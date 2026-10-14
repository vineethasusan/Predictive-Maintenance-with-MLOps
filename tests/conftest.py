import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def raw():
    from fleetpm.data import generate

    return generate(n_vehicles=150, days=330, seed=3)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, raw):
    from fleetpm.train import train

    d = tmp_path_factory.mktemp("run")
    path = d / "telemetry.csv.gz"
    raw.to_csv(path, index=False)
    card = train(str(path), str(d / "artifacts"))
    return card, d / "artifacts"
