import pytest


@pytest.fixture(autouse=True)
def _isolate_metrics(monkeypatch, tmp_path_factory):
    """Keep test telemetry out of the real data/metrics/ directory."""
    path = tmp_path_factory.mktemp("metrics")
    monkeypatch.setattr("core.metrics.METRICS_DIR", str(path))
