from collections.abc import Callable
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from typing import Literal

import polars as pl
import tenacity
from kostyl.utils import setup_logger
from openai import OpenAI
from tqdm.auto import tqdm

from .callbacks import Callback
from .callbacks import CheckpointCallback
from .structures import GenerationResult


logger = setup_logger()


class LLMTextGenerator:
    """
    Generates text responses from an LLM API using concurrent requests.

    Manages a thread pool to send batched requests to an OpenAI-compatible
    endpoint and collects results. Supports optional callbacks for checkpointing
    and other side effects during generation.
    """

    def __init__(
        self,
        api_url: str,
        key: str = "abc",
        callbacks: list[Callback] | None = None,
    ) -> None:
        """
        Initializes LLMTextGenerator.

        Args:
            api_url: Base URL of the OpenAI-compatible API endpoint.
            key: API authentication key. Defaults to ``"abc"`` for local servers
                that do not require a real key.
            callbacks: Optional list of callbacks invoked after each generation
                step and at the end of the generation loop.

        """
        self.client = OpenAI(
            api_key=key,
            base_url=api_url,
        )
        self.global_step = 0
        self.results: list[dict[str, str]] = []
        self.callbacks = callbacks
        return

    def _generate_prompt_iterator(
        self, df: pl.DataFrame, id_col: str, prompt_template: str
    ) -> Iterator[tuple[str, str]]:
        for row in df.iter_rows(named=True):
            data_id = row[id_col]
            prompt = prompt_template.format(**row)
            yield data_id, prompt
        return

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential_jitter(initial=0, max=32, jitter=1),
        reraise=True,
    )
    def _request(
        self,
        model_name: str,
        max_tokens: int,
        top_p: float,
        top_k: int,
        temperature: float,
        prompt: str,
        data_id: str,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = self.client.chat.completions.create(  # type: ignore
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        result = GenerationResult(
            data_id=data_id, prompt=prompt, response=response.choices[0].message.content
        )
        return result

    def _get_next_sample(
        self, generator: Iterator[tuple[str, str]]
    ) -> tuple[str, str] | None:
        try:
            data_id, prompt = next(generator)
            return data_id, prompt
        except StopIteration:
            return None

    def generate(  # noqa: C901
        self,
        df: pl.DataFrame,
        batch_size: int,
        id_column: str,
        model_name: str,
        prompt_template: str,
        system_prompt: str | None = None,
        top_p: float = 0.8,
        top_k: int = 40,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        response_postprocessor_fn: Callable[[GenerationResult], dict[str, str]]
        | None = None,
    ) -> pl.DataFrame:
        """
        Runs the generation loop over all rows of a DataFrame.

        Concurrently submits up to `batch_size` requests at a time, collects
        responses, and appends them to `self.results`. Invokes registered
        callbacks after every completed batch and once more at the end.

        Args:
            df: Input DataFrame. Each row is formatted into a prompt via
                `prompt_template`.
            batch_size: Maximum number of in-flight requests at any given time.
            id_column: Name of the column used as a unique row identifier stored
                in results under the ``"id"`` key.
            model_name: Name of the model to request from the API.
            prompt_template: Python format-string template. Row values are
                interpolated by column name, e.g. ``"{headline} {body}"``.
            system_prompt: Optional system message prepended to every request.
            top_p: Nucleus sampling probability threshold. Defaults to ``0.8``.
            top_k: Top-k sampling cutoff. Defaults to ``40``.
            temperature: Sampling temperature. Defaults to ``1.0``.
            max_tokens: Maximum number of tokens to generate per response.
                Defaults to ``1024``.
            response_postprocessor_fn: Optional function applied to each raw
                ``{prompt, response}`` dict before it is stored in results.

        Returns:
            A DataFrame with generated responses.

        """
        if batch_size < 1:
            raise ValueError("`batch_size` must be at least 1")

        self.results = []
        self.global_step = 0

        pg = tqdm(
            total=len(df),
            desc="Generating responses",
            unit="row",
        )
        # Initially create batch of requests
        with ThreadPoolExecutor() as executor:
            prompt_iterator = self._generate_prompt_iterator(
                df,
                id_column,
                prompt_template,
            )
            tasks_in_progress: dict[Future[GenerationResult], str] = {}
            loop_counter = 0
            while loop_counter < len(df) or len(tasks_in_progress) > 0:
                tasks2append_num = min(
                    batch_size - len(tasks_in_progress), len(df) - loop_counter
                )
                for _ in range(tasks2append_num):
                    try:
                        data_id, prompt = next(prompt_iterator)
                    except StopIteration:
                        break

                    future = executor.submit(
                        self._request,
                        data_id=data_id,
                        model_name=model_name,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        prompt=prompt,
                        system_prompt=system_prompt,
                    )
                    tasks_in_progress[future] = data_id
                    loop_counter += 1

                done, _ = wait(tasks_in_progress, return_when=FIRST_COMPLETED)

                task_to_remove: list[Future] = []
                for future in done:
                    try:
                        generation_result = future.result()
                        if response_postprocessor_fn is not None:
                            generation_result_dict = response_postprocessor_fn(
                                generation_result
                            )
                            if not isinstance(generation_result_dict, dict):
                                raise ValueError(
                                    f"`response_postprocessor_fn` must return a dict, but got {type(generation_result_dict)}"
                                )
                        else:
                            generation_result_dict = generation_result.to_dict()

                        self.results.append(generation_result_dict)
                    except Exception:
                        logger.exception(
                            f"Error processing task for data_id {tasks_in_progress[future]}"
                        )
                    task_to_remove.append(future)

                for future in task_to_remove:
                    tasks_in_progress.pop(future, None)

                if len(task_to_remove) > 0:
                    self.global_step += len(task_to_remove)
                    pg.update(len(task_to_remove))
                    self._run_callbacks(method_name="on_step")

        self._run_callbacks(method_name="on_end")
        pg.close()
        return self._build_dataframe()

    def _run_callbacks(
        self,
        method_name: Literal["on_step", "on_end"],
    ) -> None:
        if self.callbacks is not None:
            for callback in self.callbacks:
                if method_name == "on_step":
                    callback.on_step(self.results, self.global_step)
                else:
                    callback.on_end(self.results)
        return

    def _build_dataframe(self) -> pl.DataFrame:
        checkpoint_callback = None
        if self.callbacks is not None:
            for callback in self.callbacks:
                if isinstance(callback, CheckpointCallback):
                    checkpoint_callback = callback
                    break
        df = (
            checkpoint_callback.build_checkpoint_dataframe(self.results)
            if checkpoint_callback is not None
            else pl.DataFrame(self.results)
        )
        return df
