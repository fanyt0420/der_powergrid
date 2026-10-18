"""Power-flow solver implementations."""

from src.solvers.custom_pf_solver import CustomPFSolver

__all__ = ["OpenDSSSolver", "CustomPFSolver", "FiniteElementPFSolver"]


def __getattr__(name):
    if name == "OpenDSSSolver":
        from src.solvers.opendss_solver import OpenDSSSolver
        return OpenDSSSolver
    if name == "FiniteElementPFSolver":
        from src.solvers.fem_solver import FiniteElementPFSolver
        return FiniteElementPFSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
