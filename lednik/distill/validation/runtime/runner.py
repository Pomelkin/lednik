from typing import cast

import numpy as np
import pacmap
import plotly.graph_objects as go
import torch
from clearml import Task
from kostyl.utils import setup_logger
from plotly.subplots import make_subplots
from torch import Tensor

from lednik.distill.validation.structs import KNNConfig
from lednik.distill.validation.structs import LogRegConfig

from lednik.distill.validation.structs import EvaluationRunnerConfig
from lednik.distill.validation.structs import MRRConfig
from lednik.distill.validation.structs import ValidationContract
from lednik.distill.validation.metrics import calculate_logreg_metrics
from lednik.distill.validation.metrics import calculate_mrr
from lednik.distill.validation.metrics import calculate_self_exc_knn_metrics


logger = setup_logger(fmt="only_message")

_MAX_PLOTLY_POINTS = 500


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
        self.teacher_mrr_results: dict[str, float] = {}
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

        student_queries = student_embeddings[queries_mask]
        teacher_queries = teacher_embeddings[queries_mask]
        if self.config.mrr_config is not None:
            student_positives = student_embeddings[pos_mask]
            teacher_positives = teacher_embeddings[pos_mask]
            self._log_mrr(
                task=task,
                step=step,
                student_queries=student_queries,
                student_positives=student_positives,
                teacher_queries=teacher_queries,
                teacher_positives=teacher_positives,
                mrr_config=self.config.mrr_config,
            )

        if num_labels > 0:
            max_points = min(self.config.scatter_num_points, _MAX_PLOTLY_POINTS)
            teacher_scatter = teacher_queries[:max_points]
            student_scatter = student_queries[:max_points]
            labels_scatter = labels[:max_points]

            teacher_embeddings_np = teacher_scatter.cpu().float().numpy()
            student_embeddings_np = student_scatter.cpu().float().numpy()
            labels_np = labels_scatter.cpu().numpy()
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
                    teacher_embeddings=teacher_queries,
                    student_embeddings=student_queries,
                    labels=labels,
                    knn_config=self.config.knn_config,
                )
            if self.config.logreg_config is not None:
                self._log_logprob_metrics(
                    task=task,
                    step=step,
                    teacher_embeddings=teacher_queries,
                    student_embeddings=student_queries,
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

        # Reduce float precision to keep Plotly JSON small enough for ClearML API limits.
        teacher_2d_embeddings = np.round(teacher_2d_embeddings.astype(np.float32), 4)
        student_2d_embeddings = np.round(student_2d_embeddings.astype(np.float32), 4)
        labels = labels.astype(np.int32, copy=False)

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

        try:
            task.get_logger().report_plotly(
                title="Scatter Plots of 2D Embeddings",
                series="Embeddings Teacher vs Student Scatter",
                figure=fig,
                iteration=step,
            )
        except Exception as e:
            logger.warning(
                f"Failed to upload scatter plot to ClearML at step {step}: {e}"
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
                device=self.config.device,
            )
        teacher_results = self.teacher_knn_results
        student_results = calculate_self_exc_knn_metrics(
            inputs=student_embeddings,
            targets=labels,
            knn_config=knn_config,
            num_classes=num_labels,
            device=self.config.device,
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
                device=self.config.device,
            )
        teacher_logprob_results = self.teacher_logprob_results
        student_logprob_results = calculate_logreg_metrics(
            inputs=student_embeddings,
            targets=labels,
            logreg_config=logreg_config,
            num_classes=num_classes,
            device=self.config.device,
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
        teacher_queries: Tensor,
        teacher_positives: Tensor,
        student_queries: Tensor,
        student_positives: Tensor,
        mrr_config: MRRConfig,
    ) -> None:
        if len(self.teacher_mrr_results) == 0:
            self.teacher_mrr_results = calculate_mrr(
                queries=teacher_queries,
                positives=teacher_positives,
                config=mrr_config,
            )
        teacher_mrr = self.teacher_mrr_results["MRR"]
        student_mrr = calculate_mrr(
            queries=student_queries,
            positives=student_positives,
            config=mrr_config,
        )["MRR"]

        for key, value in {
            "Teacher_MRR": teacher_mrr,
            "Student_MRR": student_mrr,
        }.items():
            task.get_logger().report_scalar(
                title="Retrieval Evaluation",
                series=key,
                value=value,
                iteration=step,
            )
        return
