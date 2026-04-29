from pathlib import Path
from typing import cast

import click
from kostyl.utils import setup_logger
from redis import Redis

from lednik.distill.validation.structs import EvaluationWorkerConfig
from lednik.distill.validation.structs import ValidationContract
from lednik.distill.validation.runtime.runner import EvaluationRunner


logger = setup_logger(fmt="default")


class EvaluationWorker:
    """
    Worker class for distributed evaluation using Redis streams.

    This class manages the connection to Redis and handles evaluation tasks
    consumed from a Redis stream. It initializes the evaluation runner and
    sets up the consumer group for processing evaluation jobs.
    """

    def __init__(self, config: EvaluationWorkerConfig) -> None:
        """Initialize the evaluation worker with Redis configuration."""
        if config.redis is None:
            raise ValueError("For workers Redis configuration must be provided.")

        self.redis_client = Redis(
            host=config.redis.host,
            port=config.redis.port,
            decode_responses=False,
        )

        if not self.redis_client.ping():
            raise ValueError("Failed to connect to Redis.")

        self.stream_name = config.redis.stream_name
        self.eval_runner = EvaluationRunner(config.runner_config)
        self.group_name = "default"
        self.consumer_name = "consumer1"
        self._setup()
        return

    def _setup(self) -> None:
        if not self._stream_exists():
            logger.info(
                f"Stream '{self.stream_name}' does not exist. Creating stream..."
            )
            msg_id = self.redis_client.xadd(self.stream_name, {"init": b"1"})
            self.redis_client.xdel(self.stream_name, msg_id)  # type: ignore
            logger.info(f"Stream '{self.stream_name}' created.")

        if not self._group_exists():
            logger.info(
                f"Group '{self.group_name}' does not exist. Creating group for stream '{self.stream_name}'..."
            )
            res = self.redis_client.xgroup_create(
                self.stream_name, self.group_name, id="$", mkstream=True
            )
            logger.info(
                f"Group '{self.group_name}' created for stream '{self.stream_name}'."
            )
            if not res:
                raise ValueError("Failed to create Redis stream group.")
        return

    def _stream_exists(self) -> bool:
        try:
            self.redis_client.xinfo_stream(self.stream_name)
            return True
        except Exception:
            return False

    def _group_exists(self) -> bool:
        self.redis_client.xinfo_stream(
            self.stream_name
        )  # Ensure stream exists before checking groups
        groups_info = self.redis_client.xinfo_groups(self.stream_name)
        groups_info = cast(list[dict], groups_info)
        bytes_group_name = self.group_name.encode()
        return any(group_info["name"] == bytes_group_name for group_info in groups_info)

    def _shutdown(self) -> None:
        self.redis_client.close()
        return

    def run(self) -> None:
        """Start evaluation worker loop (msg read -> evaluate)."""
        try:
            logger.info("Evaluation worker started.")
            while True:
                response: list = self.redis_client.xreadgroup(  # type: ignore
                    groupname=self.group_name,
                    consumername=self.consumer_name,
                    streams={self.stream_name: ">"},
                    block=1000,
                )

                if len(response) == 0:
                    continue

                messages = response[0][1]
                if len(messages) == 0:
                    continue

                logger.info(f"Received {len(messages)} task(s) for evaluation.")
                processed_message_ids = []
                for message_id, message_data in messages:
                    try:
                        contract = ValidationContract.from_bytes_dict(message_data)
                        self.eval_runner.evaluate(contract)
                    except Exception as e:
                        logger.error(f"Error processing message {message_id}:\n{e}")
                    finally:
                        processed_message_ids.append(message_id)

                self.redis_client.xack(
                    self.stream_name, self.group_name, *processed_message_ids
                )
                logger.info(f"Evaluated {len(processed_message_ids)} task(s).")
        except KeyboardInterrupt:
            logger.info("Shutting down worker.")
            self._shutdown()
        return


@click.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
    help="Path to the evaluation configuration file (YAML or JSON).",
)
def main(config_path: Path) -> None:
    """Main function to start the evaluation worker."""
    config = EvaluationWorkerConfig.from_file(config_path)
    worker = EvaluationWorker(config)
    worker.run()
    return


if __name__ == "__main__":
    main()
