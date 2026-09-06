"""Phase 50-E orchestrator."""
from .cross_seed import materialize_phase50_e


def main(project_root=None):
    return materialize_phase50_e(project_root)
