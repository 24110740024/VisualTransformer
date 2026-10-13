"""Portable protocol guard. Does not impersonate the original frozen run."""
from pathlib import Path
import json

def validate_execution_protocol():
    spec=json.loads((Path(__file__).parent/'EXECUTION_PROTOCOL.json').read_text())
    if spec['protocol']!='PORTABLE_V6_20261013' or spec['checkpoint_format']!=106:
        raise ValueError('Wrong portable protocol')
    return spec

def gpu_guard():
    validate_execution_protocol()
    # Resource scheduling belongs to the caller. No cloud-job status is assumed.
