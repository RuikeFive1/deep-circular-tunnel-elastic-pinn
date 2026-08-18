from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apinn.models import MLP_APINN
from apinn.physics import equilibrium_residuals


def main() -> None:
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_APINN(hidden=40, depth=6).to(device)

    checkpoint = ROOT / "checkpoints" / "apinn_deep_elastic_paper_exact.pt"
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)

    points = torch.tensor(
        [[1.2, 0.4], [1.5, 1.0], [2.0, 1.5]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    output = model(points)
    assert output.shape == (3, 6)
    assert torch.isfinite(output).all()

    equilibrium = equilibrium_residuals(output[:, 2:], points)
    assert equilibrium.shape == (3, 2)
    assert torch.isfinite(equilibrium).all()
    print(f"smoke test passed on {device}: output={tuple(output.shape)}, equilibrium={tuple(equilibrium.shape)}")


if __name__ == "__main__":
    main()
