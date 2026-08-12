#!/usr/bin/env python3
"""Apply FT-FSOD's warn-only deterministic setting to MMEngine 0.10.7."""

from importlib.util import find_spec
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Expected MMEngine 0.10.7 source was not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


spec = find_spec("mmengine")
if spec is None or spec.origin is None:
    raise RuntimeError("MMEngine is not installed in the active environment")
root = Path(spec.origin).resolve().parent
runner_path = root / "runner" / "runner.py"
utils_path = root / "runner" / "utils.py"

runner_changed = replace_once(
    runner_path,
    """    def set_randomness(self,\n                       seed,\n                       diff_rank_seed: bool = False,\n                       deterministic: bool = False) -> None:""",
    """    def set_randomness(self,\n                       seed,\n                       diff_rank_seed: bool = False,\n                       deterministic: bool = False,\n                       warn_only: bool = True) -> None:""",
)
runner_changed |= replace_once(
    runner_path,
    """            deterministic=deterministic,\n            diff_rank_seed=diff_rank_seed)""",
    """            deterministic=deterministic,\n            diff_rank_seed=diff_rank_seed,\n            warn_only=warn_only)""",
)
utils_changed = replace_once(
    utils_path,
    """def set_random_seed(seed: Optional[int] = None,\n                    deterministic: bool = False,\n                    diff_rank_seed: bool = False) -> int:""",
    """def set_random_seed(seed: Optional[int] = None,\n                    deterministic: bool = False,\n                    diff_rank_seed: bool = False,\n                    warn_only: bool = True) -> int:""",
)
utils_changed |= replace_once(
    utils_path,
    "torch.use_deterministic_algorithms(True)",
    "torch.use_deterministic_algorithms(True, warn_only=warn_only)",
)

if runner_changed or utils_changed:
    print(f"[OK] Patched deterministic warn-only behavior in {root}")
else:
    print(f"[OK] MMEngine deterministic patch already present in {root}")
