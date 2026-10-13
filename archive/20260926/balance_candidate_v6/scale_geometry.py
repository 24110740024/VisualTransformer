"""Read-only reuse of accepted v4 geometry; do not regenerate coordinates."""
from pathlib import Path
import importlib.util
_PATH=Path(__file__).resolve().parent.parent/'scale_candidate_v4/scale_geometry.py'
_spec=importlib.util.spec_from_file_location('v4_frozen_geometry_for_v6',_PATH)
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
DIRECTORY=_module.DIRECTORY
config=_module.config
transform=_module.transform
load_scaled=_module.load_scaled
