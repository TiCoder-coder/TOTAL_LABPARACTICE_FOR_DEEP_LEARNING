"""Phase 50-D orchestrator."""
from .error_join import materialize_phase50_d


def main(project_root=None):
    return materialize_phase50_d(project_root)
