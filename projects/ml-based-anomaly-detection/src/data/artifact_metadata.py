import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


def _json_safe(value: Any, project_root: Path) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.relative_to(project_root))
        except ValueError:
            return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            k: _json_safe(v, project_root)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [_json_safe(v, project_root) for v in value]

    return value


def write_artifact_metadata(
    artifact_path: Path,
    metadata: dict[str, Any],
    project_root: Path,
) -> Path:
    artifact_path = Path(artifact_path)
    project_root = Path(project_root)

    sidecar_path = artifact_path.with_name(
        f"{artifact_path.stem}.meta.json"
    )

    payload = {
        "artifact_path": str(artifact_path.relative_to(project_root)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }

    payload = _json_safe(payload, project_root)

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    with open(sidecar_path, "w") as f:
        json.dump(payload, f, indent=2)

    return sidecar_path