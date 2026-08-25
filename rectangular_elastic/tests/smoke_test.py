from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_pinn_fields import eval_quarter_fields, load_model


def main() -> None:
    checkpoint = ROOT / "checkpoints" / "pinn_model_stageA_rectangular_gpt_v42.pth"
    model = load_model(checkpoint)
    x = np.array([0.2, 1.1, 2.5, 4.9], dtype=float)
    z = np.array([1.2, 0.9, 2.5, 4.9], dtype=float)
    fields = eval_quarter_fields(model, x, z, hard_u_bc=True, stress_param="delta")
    expected = {"ux", "uz", "sxx", "szz", "syy", "sxz"}
    assert set(fields) == expected
    for key, values in fields.items():
        assert values.shape == (4,), (key, values.shape)
        assert np.isfinite(values).all(), key
    print("rectangular elastic smoke test passed: checkpoint load and six-field inference are finite")


if __name__ == "__main__":
    main()
