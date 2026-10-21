"""Load the SALAD backend copied/cloned INSIDE this standalone project.

The surrounding original WORKSPACE project is never searched/imported.
"""
from pathlib import Path
import os, sys

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def salad_dir() -> Path:
    override = os.environ.get('WEATHERVPR_SALAD_SOURCE')
    base = Path(override).expanduser() if override else (
        project_root() / 'third_party' / 'salad')
    base = base.resolve()
    if not (base / 'workflow' / 'model.py').is_file():
        raise FileNotFoundError(
            f'SALAD code missing: {base}/workflow/model.py. '
            'Run bash scripts/setup_sources.sh first; or set WEATHERVPR_SALAD_SOURCE '
            'to a standalone salad source folder.')
    return base

def enable_backend_imports() -> Path:
    backend = salad_dir()
    if str(backend) not in sys.path:
        sys.path.insert(0,str(backend))
    return backend
