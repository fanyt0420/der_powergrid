"""Solver-independent QSTS simulation with a unified DER interface."""

from src.der_model import DERModel, PVModel, WindModel, BESSModel, EVModel
from src.power_flow import DERCommand, NetworkModel, OperatingPoint, PFResult, PFState, PowerFlowSolver, SolverContext
from src.qsts import run_qsts

__all__ = [
    "DERModel",
    "PVModel",
    "WindModel",
    "BESSModel",
    "EVModel",
    "OpenDSSSolver",
    "FiniteElementPFSolver",
    "DERCommand",
    "NetworkModel",
    "OperatingPoint",
    "PFResult",
    "PFState",
    "PowerFlowSolver",
    "SolverContext",
    "run_qsts",
]


def __getattr__(name):
    if name in {"OpenDSSSolver", "FiniteElementPFSolver"}:
        from src import solvers
        return getattr(solvers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
