"""Participant-disjoint dataset splits.

Splitting individual frames at random would put frames from the same person --
often from the same footstep -- on both sides of the split. A model can then
score well by recognising the participant, and the reported number means
nothing. Every split here is by participant, and the split is written to JSON so
results are reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_DIR = REPO_ROOT / "data" / "splits"


@dataclass(frozen=True)
class Split:
    name: str
    train: list[str]
    val: list[str]
    test: list[str]
    seed: int
    note: str = ""

    def __post_init__(self) -> None:
        groups = {"train": self.train, "val": self.val, "test": self.test}
        seen: dict[str, str] = {}
        for gname, members in groups.items():
            for m in members:
                if m in seen:
                    raise ValueError(
                        f"participant {m} appears in both {seen[m]} and {gname} -- "
                        "this leaks identity across the split"
                    )
                seen[m] = gname

    @property
    def all_participants(self) -> list[str]:
        return sorted(self.train + self.val + self.test, key=_pnum)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else DEFAULT_SPLIT_DIR / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @staticmethod
    def load(path: str | Path) -> "Split":
        return Split(**json.loads(Path(path).read_text()))


def _pnum(p: str) -> int:
    return int(p.lstrip("P"))


def participant_split(participants: Sequence[str], *, name: str = "participant_holdout",
                      val_frac: float = 0.2, test_frac: float = 0.2,
                      seed: int = 0) -> Split:
    """Deterministic participant-level train/val/test split."""
    import random

    ps = sorted(set(participants), key=_pnum)
    if len(ps) < 3:
        raise ValueError("need at least 3 participants to make three groups")
    rng = random.Random(seed)
    shuffled = ps[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(len(ps) * test_frac))
    n_val = max(1, round(len(ps) * val_frac))
    if n_test + n_val >= len(ps):
        raise ValueError("val+test fractions leave no training participants")
    test = sorted(shuffled[:n_test], key=_pnum)
    val = sorted(shuffled[n_test:n_test + n_val], key=_pnum)
    train = sorted(shuffled[n_test + n_val:], key=_pnum)
    return Split(name=name, train=train, val=val, test=test, seed=seed,
                 note="Participant-disjoint. Test participants are unseen during training.")


def leave_one_subject_out(participants: Sequence[str], *, seed: int = 0) -> list[Split]:
    """One split per participant: that participant is the test set."""
    ps = sorted(set(participants), key=_pnum)
    splits = []
    for i, held in enumerate(ps):
        rest = [p for p in ps if p != held]
        val = [rest[i % len(rest)]]
        train = [p for p in rest if p not in val]
        splits.append(Split(name=f"loso_{held}", train=train, val=val, test=[held],
                            seed=seed, note=f"Leave-one-subject-out; test = {held}."))
    return splits
