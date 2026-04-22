from typing import cast

import numpy as np
import pacmap
import plotly.graph_objects as go
import torch
from clearml import Task
from kostyl.utils import setup_logger
from plotly.subplots import make_subplots
from torch import Tensor

from lednik.distill.validation.configs import KNNConfig
from lednik.distill.validation.configs import LogRegConfig

from .configs import EvaluationRunnerConfig
from .configs import MRRConfig
from .contracts import ValidationContract
from .metrics import calculate_logreg_metrics
from .metrics import calculate_mrr
from .metrics import calculate_self_exc_knn_metrics


logger = setup_logger(fmt="only_message")


class EvaluationRunner:
    """
    Runs evaluation routines for teacher and student embeddings.

    The runner computes and logs retrieval and classification-oriented metrics,
    including 2D scatter visualization, MRR, KNN metrics, and optional logistic
    probing.
    """

    def __init__(self, config: EvaluationRunnerConfig) -> None:
        """Initialize evaluation configuration, caches, and async probing state."""
        self.config = config

        self.teacher_knn_results: dict[str, float] = {}
        self.teacher_logprob_results: dict[str, float] = {}
        self.teacher_2d_embeddings: np.ndarray | None = None
        self.task: Task | None = None
        return

    def evaluate(
        self,
        validation_contract: ValidationContract,
    ) -> None:
        """Run evaluation and log metrics for the current validation step.

        Logs:
        - 2D scatter plots for teacher and student embeddings.
        - MRR retrieval metric when retrieval evaluation is enabled.
        - KNN F1/Accuracy when num_labels > 0.
        - Logistic probing metrics asynchronously when num_labels > 0.

        Args:
            validation_contract: The contract containing all necessary data for evaluation.
        """
        if self.task is None or self.task.id != validation_contract.task_id:
            self.task = Task.get_task(task_id=validation_contract.task_id)

        task = cast(Task, self.task)
        step = validation_contract.current_step
        teacher_embeddings = validation_contract.teacher_embeddings
        student_embeddings = validation_contract.student_embeddings
        queries_mask = validation_contract.queries_mask
        pos_mask = validation_contract.pos_mask
        num_labels = validation_contract.num_classes
        labels = validation_contract.labels

        if self.config.mrr_config is not None:
            queries_emb = student_embeddings[queries_mask]
            pos_emb = student_embeddings[pos_mask]
            self._log_mrr(
                task=task,
                step=step,
                queries=queries_emb,
                positives=pos_emb,
                mrr_config=self.config.mrr_config,
            )

        if num_labels > 0:
            teacher_embeddings_np = (
                teacher_embeddings[: self.config.scatter_num_points]
                .float()
                .cpu()
                .numpy()
            )
            student_embeddings_np = (
                student_embeddings[: self.config.scatter_num_points]
                .float()
                .cpu()
                .numpy()
            )
            labels_np = labels[: self.config.scatter_num_points].float().cpu().numpy()
            self._log_embeddings_scatter(
                task=task,
                step=step,
                teacher_embeddings=teacher_embeddings_np,
                student_embeddings=student_embeddings_np,
                labels=labels_np,
            )

            if self.config.knn_config is not None:
                self._log_knn_metrics(
                    task=task,
                    step=step,
                    num_labels=num_labels,
                    teacher_embeddings=teacher_embeddings,
                    student_embeddings=student_embeddings,
                    labels=labels,
                    knn_config=self.config.knn_config,
                )
            if self.config.logreg_config is not None:
                self._log_logprob_metrics(
                    task=task,
                    step=step,
                    teacher_embeddings=teacher_embeddings,
                    student_embeddings=student_embeddings,
                    labels=labels,
                    num_classes=num_labels,
                    logreg_config=self.config.logreg_config,
                )
        return

    def _log_embeddings_scatter(
        self,
        task: Task,
        step: int,
        teacher_embeddings: np.ndarray,
        student_embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:

        pmap = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)

        if self.teacher_2d_embeddings is None:
            teacher_2d_embeddings = pmap.fit_transform(teacher_embeddings, init="pca")
            self.teacher_2d_embeddings = teacher_2d_embeddings
        else:
            teacher_2d_embeddings = self.teacher_2d_embeddings

        student_2d_embeddings = pmap.fit_transform(student_embeddings, init="pca")

        teacher_2d_embeddings = cast(np.ndarray, teacher_2d_embeddings)
        student_2d_embeddings = cast(np.ndarray, student_2d_embeddings)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Teacher embeddings", "Student embeddings"),
            horizontal_spacing=0.08,
        )
        fig.add_trace(
            go.Scatter(
                x=teacher_2d_embeddings[:, 0].tolist(),
                y=teacher_2d_embeddings[:, 1].tolist(),
                mode="markers",
                marker={
                    "color": labels.tolist(),
                    "colorscale": "Plotly3",
                    "showscale": True,
                    "opacity": 0.7,
                },
                name="Teacher",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=student_2d_embeddings[:, 0].tolist(),
                y=student_2d_embeddings[:, 1].tolist(),
                mode="markers",
                marker={
                    "color": labels.tolist(),
                    "colorscale": "Plotly3",
                    "showscale": False,
                    "opacity": 0.7,
                },
                name="Student",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="dim0", row=1, col=1)
        fig.update_yaxes(title_text="dim1", row=1, col=1)
        fig.update_xaxes(title_text="dim0", row=1, col=2)
        fig.update_yaxes(title_text="dim1", row=1, col=2)

        task.get_logger().report_plotly(
            title="Scatter Plots of 2D Embeddings",
            series="Embeddings Teacher vs Student Scatter",
            figure=fig,
            iteration=step,
        )
        return

    def _log_knn_metrics(
        self,
        task: Task,
        step: int,
        num_labels: int,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor,
        labels: torch.Tensor,
        knn_config: KNNConfig,
    ) -> None:
        if len(self.teacher_knn_results) == 0:
            self.teacher_knn_results = calculate_self_exc_knn_metrics(
                inputs=teacher_embeddings,
                targets=labels,
                knn_config=knn_config,
                num_classes=num_labels,
            )
        teacher_results = self.teacher_knn_results
        student_results = calculate_self_exc_knn_metrics(
            inputs=student_embeddings,
            targets=labels,
            knn_config=knn_config,
            num_classes=num_labels,
        )

        metrics = {
            "Teacher_F1": teacher_results["F1"],
            "Teacher_Accuracy": teacher_results["Accuracy"],
            "Student_F1": student_results["F1"],
            "Student_Accuracy": student_results["Accuracy"],
        }
        for key, value in metrics.items():
            task.get_logger().report_scalar(
                title="KNN Evaluation Metrics",
                series=key,
                value=value,
                iteration=step,
            )
        return

    def _log_logprob_metrics(
        self,
        task: Task,
        step: int,
        teacher_embeddings: Tensor,
        student_embeddings: Tensor,
        labels: Tensor,
        num_classes: int,
        logreg_config: LogRegConfig,
    ) -> None:
        if len(self.teacher_logprob_results) == 0:
            self.teacher_logprob_results = calculate_logreg_metrics(
                inputs=teacher_embeddings,
                targets=labels,
                logreg_config=logreg_config,
                num_classes=num_classes,
            )
        teacher_logprob_results = self.teacher_logprob_results
        student_logprob_results = calculate_logreg_metrics(
            inputs=student_embeddings,
            targets=labels,
            logreg_config=logreg_config,
            num_classes=num_classes,
        )

        metrics = {
            "Teacher_Accuracy": teacher_logprob_results["Accuracy"],
            "Teacher_F1": teacher_logprob_results["F1"],
            "Student_Accuracy": student_logprob_results["Accuracy"],
            "Student_F1": student_logprob_results["F1"],
        }
        for key, value in metrics.items():
            task.get_logger().report_scalar(
                title="Logistic Regression Evaluation Metrics",
                series=key,
                value=value,
                iteration=step,
            )
        return

    def _log_mrr(
        self,
        task: Task,
        step: int,
        queries: Tensor,
        positives: Tensor,
        mrr_config: MRRConfig,
    ) -> None:
        mrr_results = calculate_mrr(
            queries=queries,
            positives=positives,
            config=mrr_config,
        )
        mrr = mrr_results["MRR"]

        task.get_logger().report_scalar(
            title="Retrieval Evaluation",
            series="MRR",
            value=mrr,
            iteration=step,
        )
        return
