from dataclasses import dataclass

from torch import Tensor

from .generic import bytes_to_tensor
from .generic import tensor_to_bytes


@dataclass
class ValidationContract:
    """A contract for validation data used in the distillation process."""

    task_id: str
    current_step: int
    teacher_embeddings: Tensor
    student_embeddings: Tensor
    queries_mask: Tensor
    pos_mask: Tensor
    labels: Tensor
    num_classes: int

    def to_bytes_dict(self) -> dict[bytes, bytes]:
        """Serializes the ValidationContract to a dictionary of bytes."""
        return {
            b"task_id": self.task_id.encode("utf-8"),
            b"current_step": self.current_step.to_bytes(
                4, byteorder="little", signed=True
            ),
            b"teacher_embeddings": tensor_to_bytes(self.teacher_embeddings),
            b"student_embeddings": tensor_to_bytes(self.student_embeddings),
            b"queries_mask": tensor_to_bytes(self.queries_mask),
            b"pos_mask": tensor_to_bytes(self.pos_mask),
            b"labels": tensor_to_bytes(self.labels),
            b"num_classes": self.num_classes.to_bytes(
                4, byteorder="little", signed=True
            ),
        }

    @classmethod
    def from_bytes_dict(cls, bytes_dict: dict[bytes, bytes]) -> "ValidationContract":
        """Deserializes a ValidationContract from a dictionary of bytes."""
        return cls(
            task_id=bytes_dict[b"task_id"].decode("utf-8"),
            teacher_embeddings=bytes_to_tensor(bytes_dict[b"teacher_embeddings"]),
            student_embeddings=bytes_to_tensor(bytes_dict[b"student_embeddings"]),
            queries_mask=bytes_to_tensor(bytes_dict[b"queries_mask"]),
            pos_mask=bytes_to_tensor(bytes_dict[b"pos_mask"]),
            labels=bytes_to_tensor(bytes_dict[b"labels"]),
            num_classes=int.from_bytes(
                bytes_dict[b"num_classes"], byteorder="little", signed=True
            ),
            current_step=int.from_bytes(
                bytes_dict[b"current_step"], byteorder="little", signed=True
            ),
        )
