from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Data class to store the result of a single text generation."""

    data_id: str
    prompt: str
    response: str

    def to_dict(self) -> dict[str, str]:
        """Convert the GenerationResult to a dictionary."""
        return {
            "id": self.data_id,
            "prompt": self.prompt,
            "response": self.response,
        }
