from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def signed_correlation() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.5, -0.25],
            [0.5, 1.0, 0.0],
            [-0.25, 0.0, 1.0],
        ]
    )


@pytest.fixture
def config_writer(tmp_path: Path):
    def write(
        input_path: Path,
        *,
        source_type: str = "matrix",
        semantics: str = "correlation",
        extra_input: str = "",
        extra_root: str = "",
    ) -> Path:
        config_path = tmp_path / f"config_{len(list(tmp_path.glob('config_*.yaml')))}.yaml"
        config_path.write_text(
            "input:\n"
            f"  source_type: {source_type}\n"
            f"  path: {input_path.as_posix()}\n"
            f"  matrix_semantics: {semantics}\n"
            f"{extra_input}"
            f"{extra_root}",
            encoding="utf-8",
        )
        return config_path

    return write
