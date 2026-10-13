# 最新 main 与有限元求解器兼容性说明

本次以 `origin/main` 的 2026-10-13 版本为基线移植原有限元代码，保留了当前主线的 `NetworkModel`、`ProfileStore`、QSTS 和 OpenDSS 实现。

原来基于早期 `main` 的 `CustomPFSolver` 不能直接覆盖最新接口：最新 `OperatingPoint` 使用时间戳而不是小时编号；`PFResult` 必须返回最终 DER 命令、控制迭代与潮流求解次数；QSTS 需要节点、系统和支路三组结果。直接替换旧 `run.py` 还会绕开 `ProfileStore`。因此移植的线单元求解器单独放在 `src/solvers/fem_solver.py`，并以 `--solver fem` 接入最新 `run.py`，保留默认 OpenDSS 路径。

主线 `ieee13_measured` 与 `ieee13_unbalanced_der` 的 `network.dss` 原先指向开发者机器上的绝对路径，在本机无法运行。现改为相对路径，同时更新两个数据生成脚本，避免重新生成时恢复错误路径。修正后两个场景的默认 OpenDSS QSTS 均已完整跑通（分别为 96 个与 24 个时间步）。

验证包括：最新 QSTS 接口上的含负荷单时间步、线路长度单位检查、完整 IEEE 13 固定抽头电压回归，均通过。完整 IEEE 13 的结果仍由单独的 `run_ieee13_fem.py` 生成：38 个相节点、9 条路径，最大电压幅值误差约 0.000156 pu，线路首端最大复电流差约 0.479 A。该结果不代表线单元 QSTS 后端已支持 IEEE 13 的全部设备或已实现加速。
