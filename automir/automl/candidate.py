import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CandidateConfig:
    """Complete specification of an AutoML model and training configuration."""

    # Unique identifiers
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation: int = 0

    # Audio representations
    representation: str = "logmel"  # 'logmel', 'tempogram', 'logmel_tempogram'
    segment_duration: float = 8.0  # 4.0, 8.0, 12.0
    n_mels: int = 96  # 64, 96, 128

    # Architecture
    conv_blocks: int = 3  # 2, 3, 4
    base_channels: int = 32  # 16, 32, 64
    kernel_size: int = 3  # 3, 5, 7
    use_gru: bool = False
    gru_hidden: int = 64  # 32, 64, 128
    dropout: float = 0.25  # 0.10 - 0.50

    # Training Hyperparameters
    learning_rate: float = 1e-3  # 1e-4 to 3e-3
    weight_decay: float = 1e-4  # 1e-6 to 1e-3
    batch_size: int = 32  # 16, 32, 64

    # Optimization tracking
    rank: int = 0
    crowding_distance: float = 0.0
    failed: bool = False
    failure_reason: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate candidate hyperparameters against acceptable bounds."""
        if self.representation not in ["logmel", "tempogram", "logmel_tempogram"]:
            return False
        if self.segment_duration not in [4.0, 8.0, 12.0]:
            return False
        if self.n_mels not in [64, 96, 128]:
            return False
        if self.conv_blocks not in [2, 3, 4]:
            return False
        if self.base_channels not in [16, 32, 64]:
            return False
        if self.kernel_size not in [3, 5, 7]:
            return False
        if self.gru_hidden not in [32, 64, 128]:
            return False
        if not (0.05 <= self.dropout <= 0.60):
            return False
        if not (1e-5 <= self.learning_rate <= 1e-2):
            return False
        if not (1e-7 <= self.weight_decay <= 1e-2):
            return False
        if self.batch_size not in [16, 32, 64]:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateConfig":
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
