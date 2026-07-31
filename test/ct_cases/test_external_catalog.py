from __future__ import annotations

import json
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def test_external_real_data_catalog_is_non_downloading_and_actionable():
    payload = json.loads(
        (DATA_ROOT / "external_catalog.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "1.0"
    assert {record["dataset_id"] for record in payload["datasets"]} >= {
        "2detect",
        "walnut_ct",
    }
    for record in payload["datasets"]:
        assert record["bundled"] is False
        assert record["homepage"].startswith("https://")
        assert record["local_root_env"].startswith("INVFRAMEWORK_")
        assert "external" in record["capability_tags"]
        assert record["license"]
