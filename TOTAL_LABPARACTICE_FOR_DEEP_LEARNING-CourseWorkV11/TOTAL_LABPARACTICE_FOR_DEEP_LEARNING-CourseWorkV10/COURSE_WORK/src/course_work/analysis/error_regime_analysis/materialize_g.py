"""Phase 50-G orchestrator."""
from .finalization import materialize_phase50_g


def main(project_root=None):
    return materialize_phase50_g(project_root)
