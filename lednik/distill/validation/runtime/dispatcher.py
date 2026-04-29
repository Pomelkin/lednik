from kostyl.utils import setup_logger
from redis import Redis

from lednik.distill.validation.structs import RedisConfig
from lednik.distill.validation.structs import ValidationContract
from lednik.distill.validation.runtime.runner import EvaluationRunner


logger = setup_logger(fmt="only_message")


class EvaluationDispatcher:
    """Dispatcher for validation contracts, responsible for sending them to Redis or executing locally."""

    def __init__(
        self,
        redis_config: RedisConfig | None = None,
        evaluation_runner: EvaluationRunner | None = None,
    ) -> None:
        """Initializes the EvaluationDispatcher."""
        if redis_config is None and evaluation_runner is None:
            raise ValueError(
                "At least one of redis_config or evaluation_runner must be provided for:"
                "\n- If redis_config is not provided, evaluation_runner must be provided for local evaluation."
                "\n- If evaluation_runner is not provided, redis_config must be provided for remote evaluation."
                "\n- If both are provided, the dispatcher will attempt to use Redis for dispatching and fall back to local evaluation if Redis is unavailable."
            )
        if redis_config is not None:
            client = Redis(
                host=redis_config.host,
                port=redis_config.port,
                decode_responses=False,
            )
            stream_name = redis_config.stream_name

            if not client.ping():
                client = None
                logger.warning(
                    f"Could not connect to Redis server at {redis_config.host}:{redis_config.port}. "
                    "Evaluation tasks will be executed locally."
                )
        else:
            client = None
            stream_name = "default"

        self.evaluation_runner = evaluation_runner
        self.redis_client = client
        self.stream_name = stream_name
        return

    def dispatch(self, contract: ValidationContract) -> None:
        """
        Dispatch a validation contract for evaluation.

        Attempts to send the validation contract to Redis for remote evaluation.
        If Redis is not configured or the dispatch fails, falls back to local evaluation.

        Args:
            contract (ValidationContract): The validation contract to be evaluated.

        Returns:
            None
        """
        if self.redis_client is not None:
            try:
                self.redis_client.xadd(
                    self.stream_name,
                    contract.to_bytes_dict(),  # type: ignore
                )
                return
            except Exception as e:
                logger.warning(
                    f"Failed to dispatch validation contract to Redis: {e}"
                    "Evaluation will be executed locally."
                )
                self.redis_client = (
                    None  # Disable Redis client to avoid future attempts
                )

        # If Redis is not configured or message publication fails, we need to fallback to local evaluation.
        if self.evaluation_runner is None:
            raise ValueError(
                "Due to impossibility of sending data to a remote worker (Redis is not configured or unavailable),"
                " local evaluation is required, but no evaluation runner is configured."
                " Please provide an EvaluationRunner instance to the EvaluationDispatcher."
            )
        self.evaluation_runner.evaluate(contract)
        return
