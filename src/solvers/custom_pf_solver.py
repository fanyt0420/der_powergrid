"""Reserved location for the future native three-phase power-flow solver."""

from __future__ import annotations

from src.power_flow import NetworkModel, OperatingPoint, PFResult, PFState, SolverContext


class CustomPFSolver:
    """Skeleton for a solver that implements the PowerFlowSolver contract.

    This class intentionally has no algorithm yet. Implement ``build`` to preprocess
    NetworkModel data (for example bus-phase indices, topology, or Ybus), then
    implement ``solve`` to return PFResult using the supplied OperatingPoint and
    previous PFState.
    """

    def build(self, network: NetworkModel) -> None:
        self.network = network

    def solve(
        self,
        operating_point: OperatingPoint,
        state: PFState | None,
        context: SolverContext | None = None,
    ) -> PFResult:
        raise NotImplementedError("CustomPFSolver has not been implemented yet.")
