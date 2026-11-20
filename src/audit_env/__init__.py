"""Public package surface for FinRepGYM."""

from .env import AuditEnv, ContractViolation, safe_calculate
from .generator import DEFAULT_CONFIG, ISSUE_TYPES, SCHEMA_VERSION, generate, load_episode, save_episode
from .matcher import canonical_ids, compute_metrics, match, normalize_finding, validate_submission
from .reward import terminal_reward
from .transforms import permute_episode, remap_submission

__all__ = [
    "AuditEnv", "ContractViolation", "safe_calculate",
    "DEFAULT_CONFIG", "ISSUE_TYPES", "SCHEMA_VERSION", "generate", "load_episode", "save_episode",
    "canonical_ids", "compute_metrics", "match", "normalize_finding", "validate_submission",
    "terminal_reward", "permute_episode", "remap_submission",
]
__version__ = "0.1.0-skeleton"
