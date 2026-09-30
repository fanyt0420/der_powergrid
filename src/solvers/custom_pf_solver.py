"""Reserved location for the future native three-phase power-flow solver."""

from __future__ import annotations

from src.power_flow import NetworkModel, OperatingPoint, PFResult, PFState, SolverContext


class CustomPFSolver:
    """Skeleton for a solver that implements the PowerFlowSolver contract.

    This class intentionally has no algorithm yet. Implement ``build`` to preprocess
    NetworkModel data (for example bus-phase indices, topology, or Ybus), then
    implement ``solve`` to return PFResult using the supplied OperatingPoint and
    previous PFState.

    The contract already carries the DER control laws: each
    :class:`~src.power_flow.DERCommand` includes ``controller_type`` and its
    parameters (Vref, droop, Qmin/Qmax, ...), and exposes
    :meth:`~src.power_flow.DERCommand.adjusted_q` so a native solver can add the
    algebraic control equations ``Q = g(V)`` directly to its Newton system instead
    of using the fixed-point loop required by OpenDSS. The resulting final commands
    must be reported in ``PFResult.final_der_commands`` and the number of control
    updates in ``PFResult.control_iterations``.
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
