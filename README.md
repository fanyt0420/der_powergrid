# OpenDSS DER QSTS 框架

本项目支持多个独立算例。网络拓扑、负荷、DER 设备、DER 时序和输出均按算例目录组织；`run.py` 不默认任何特定网络规模、节点名称或 DER 数量。

## 算例目录

```text
data/
├── generate_data/
│   ├── three_node.py       # 3 节点示例数据生成器
│   └── radial_12bus.py     # 12 节点、多支路、多 DER 合成馈线数据生成器
├── three_node/             # three_node.py 的生成结果
└── radial_12bus/           # radial_12bus.py 的生成结果

output/
├── three_node/
└── radial_12bus/
```

每个算例目录均包含：

```text
<case>/
├── network.json            # 规范网络数据，供自研求解器读取
├── network.dss             # 同一网络的 OpenDSS 表示
├── load_profiles.csv       # 逐负荷 P/Q 时序
└── der_profiles/           # 每台 DER 一份时序和控制器配置
```

`network.json` 是固定网络的规范输入，包含母线、支路、三相阻抗矩阵、设备类型、接入节点、额定参数和 DER profile 相对路径。`network.dss` 是 OpenDSS 后端的兼容文件，由同一个算例生成器创建。

## 生成和运行

安装依赖：

```powershell
pip install -r requirements.txt
```

生成并运行三节点算例：

```powershell
python data\generate_data\three_node.py
python run.py --data-dir data\three_node
```

生成并运行更复杂的多支路算例：

```powershell
python data\generate_data\radial_12bus.py
python run.py --data-dir data\radial_12bus
```

默认结果写入 `output/<算例目录名>/`。如需指定位置：

```powershell
python run.py --data-dir data\radial_12bus --output output\experiment_a
```

`radial_12bus` 是合成的三相 12 节点径向多支路馈线，而非 IEEE 123 节点基准。它包含 12 个基础负荷、12 条支路、2 台 PV、1 台风机、1 台 BESS 和 1 个 EV，用于验证当前通用数据接口和多 DER QSTS 能否处理复杂于三节点的拓扑。

## 网络与时序接口

负荷 profile 是长表。每一个时刻必须包含 `network.json` 中所有 `kind="load"` 设备各一行：

```csv
hour,load_name,p_kw,q_kvar
0,Load1,270.630,90.210
0,Load2,217.124,70.246
```

每台 DER 的 `profile_file` 由 `network.json` 内设备定义指定；因此 `run.py` 会自动发现并创建 PV、Wind、BESS、EV 模型，不再写死 `PV1`、`bus1` 或三个负荷。

所有 DER profile 必须包含 `hour` 和 `controller_type`。当前支持：

| 控制器 | 行为 |
| --- | --- |
| `constant_pq` | 直接采用该时刻 profile 的 P/Q |
| `volt_var` | 固定 profile P，按接入母线电压迭代调整 Q |
| `soc_schedule` | 采用计划 BESS P/Q，但按 SOC 边界裁剪实际 P |

## 求解器接口

公共接口位于 `src/power_flow.py`：

| 类型 | 职责 |
| --- | --- |
| `NetworkModel` | 固定网络数据，从 `network.json` 读取 |
| `OperatingPoint` | 某时刻全部负荷复功率与 DER 命令 |
| `DERCommand` | 单台 DER 的 P/Q、控制器类型和参数 |
| `PFState` | 上一时刻复电压初值与求解器状态 |
| `PFResult` | 电压、线路结果、系统汇总、迭代次数、下一状态 |
| `SolverContext` | 时间步、控制迭代编号、上一轮结果、未来联立方程入口 |

```python
class PowerFlowSolver(Protocol):
    def build(self, network: NetworkModel) -> None: ...

    def solve(
        self,
        operating_point: OperatingPoint,
        state: PFState | None,
        context: SolverContext | None = None,
    ) -> PFResult: ...
```

当前后端为 `src/solvers/opendss_solver.py` 中的 `OpenDSSSolver`。未来在 `src/solvers/custom_pf_solver.py` 实现 `CustomPFSolver` 后，只需在运行入口替换求解器实例；QSTS、DER 控制器和数据格式无需修改。

## QSTS、控制和状态传递

每小时执行：

```text
读取当前逐负荷 P/Q 与各 DER profile
→ 构造 OperatingPoint
→ PowerFlowSolver.solve(OperatingPoint, PFState)
→ 按 PFResult 电压更新 Volt-VAR 控制器
→ P/Q 改变则再次 solve，直到外层控制收敛
→ BESS 推进 SOC，PFResult.next_state 传给下一时刻
```

PV Volt-VAR 使用：

```text
Q = clip[k × (Vref − Vbus), −Qmax, Qmax]
```

BESS 的 SOC 为跨时段显式状态；其容量、初始 SOC、SOC 下限和效率从 `network.json` 的 BESS 设备定义读取。PV、Wind、EV 是无状态 profile 回放模型。`PFState` 已传递给下一时刻，为自研求解器提供电压 warm start 接口。

## 输出

每个算例输出目录包含：

- `qsts_bus_voltages.csv`：每时刻、每母线相别的电压幅值和相角；
- `qsts_system.csv`：总负荷、外层控制迭代次数、潮流内部迭代次数、电压范围、损耗、源端功率、最终 DER P/Q 和 BESS SOC。

## DER 模块详解

### 统一 DER 接口

所有 DER 均继承 `src/der_model.py` 中的 `DERModel`。QSTS 不针对 PV、风机、储能或 EV 写类型判断；它只在每个时刻向模型请求注入命令、根据潮流结果更新控制器，并在收敛后推进状态。

```python
class DERModel:
    def _validate_profile(self) -> None:
        """校验设备时序和 controller_type。"""

    def get_injection(self, time_step: int) -> dict[str, object]:
        """返回计划 P/Q、控制器类型和参数。"""

    def to_command(self, injection: dict[str, object]) -> DERCommand:
        """将模型数据转为求解器无关的 DERCommand。"""

    def control_step(self, pf_result: PFResult, injection: dict[str, object]) -> dict[str, object]:
        """根据当前潮流结果更新当前时刻控制命令。"""

    def advance_state(self, injection: dict[str, object], dt_hours: float) -> None:
        """将本时刻状态传递到下一时刻。"""
```

`get_injection()`、`to_command()` 和 `control_step()` 处理同一时刻的代数关系；`advance_state()` 只在当前时刻收敛后调用，用于跨时段物理状态。

### 当前 DER 类型

| 模型 | OpenDSS 元件 | P 的符号 | 控制类型 | 跨时段状态 |
| --- | --- | --- | --- | --- |
| `PVModel` | `Generator` | 正值为发电 | `volt_var` 或 `constant_pq` | 无 |
| `WindModel` | `Generator` | 正值为发电 | `constant_pq` | 无 |
| `BESSModel` | `Storage` | 正值放电、负值充电 | `soc_schedule` | SOC |
| `EVModel` | `Load` | 正值为充电负荷 | `constant_pq` | 无 |

DER 设备的名称、种类、接入母线、额定功率、储能容量和 profile 文件均由 `network.json` 的 `devices` 读取。例如：

```json
{
  "id": "BESS1",
  "kind": "bess",
  "bus": "b6",
  "p_rated_kw": 80,
  "energy_kwh": 500,
  "reserve_soc": 20,
  "initial_soc": 55,
  "profile_file": "der_profiles/bess1.csv"
}
```

这也是 `run.py` 不再写死特定 DER 或节点名称的原因。

### DER profile 示例

PV 的 Volt-VAR profile：

```csv
hour,pv_kw,pv_kvar,controller_type,q_limit_kvar,v_ref_pu,droop_kvar_per_pu
12,180,0,volt_var,45,1.0,540
```

BESS 的计划功率 profile：

```csv
hour,bess_kw,bess_kvar,controller_type
10,-45,0,soc_schedule
18,35,0,soc_schedule
```

`controller_type` 必须逐时给出，因此后续可在同一台设备的不同运行时段切换控制方式。当前实现支持 `constant_pq`、`volt_var` 和 `soc_schedule`；不支持的类型会在 profile 读取时明确报错。

### PV Volt-VAR 方程

PV 在潮流求解后读取其接入母线所有相别的平均标幺电压 \(V_{bus}\)，并计算：

\[
Q_{PV}=\operatorname{clip}\left[k(V_{ref}-V_{bus}),-Q_{max},Q_{max}\right]
\]

低电压时注入正无功，高电压时吸收无功。若新 Q 与上一轮不同，QSTS 将生成新 `DERCommand` 并再次调用同一求解器。这个控制器是代数、无状态的：下一时刻重新由该时刻电压计算。

### BESS SOC 状态方程

BESS 在写入求解器前先根据当前 SOC 约束其计划功率，保证不会突破最小 SOC 或 100% SOC。收敛后，以最终实际功率更新：

\[
SOC_{t+1}=SOC_t-\frac{P_t\Delta t}{\eta_{dis}E}\times100,\quad P_t>0
\]

\[
SOC_{t+1}=SOC_t+\frac{|P_t|\eta_{ch}\Delta t}{E}\times100,\quad P_t<0
\]

其中 \(E\)、SOC 下限、初始 SOC 和效率从该 BESS 的网络设备定义读取，\(\Delta t\) 当前为 1 小时。`qsts_system.csv` 的 `<bess_name>_soc_pct` 是每个时刻末、将传给下一时刻的 SOC。

## QSTS 求解过程详解

对每一个 `hour`，QSTS 的实际顺序为：

```text
1. 从 load_profiles.csv 读取所有基础负荷的 P/Q
2. 从各 DER profile 读取计划 P/Q、controller_type 与控制参数
3. 构造 OperatingPoint，并携带上一时刻 PFState 调用求解器
4. 获取 PFResult：节点复电压、线路结果、系统汇总
5. 调用每台 DER 的 control_step(PFResult, ...)
6. 若任一 DER 的 P/Q 改变，重新构造 OperatingPoint 并再次求解
7. 当 P/Q 不再改变时，推进 BESS 等跨时段状态
8. 保存电压、最终 DER 命令、SOC、损耗和迭代信息
```

对于网络方程，求解器在给定负荷和 DER 注入时求解三相非线性潮流：

\[
\frac{S_i^*}{V_i^*}=\sum_jY_{ij}V_j
\]

当前 `OpenDSSSolver` 用 OpenDSS 完成该方程组求解。PV Volt-VAR 控制方程不直接嵌入 OpenDSS 的内部牛顿迭代，而通过“潮流 → 控制器 → 潮流”的外层固定点迭代与电网方程耦合。`control_iterations` 记录外层控制更新次数，`pf_iterations` 记录 OpenDSS 单次潮流的内部迭代次数。

## 状态传递详解

当前有两类时序状态：

| 状态 | 保存位置 | 从 \(t\) 到 \(t+1\) 的传递方式 |
| --- | --- | --- |
| 节点复电压 | `PFResult.next_state.voltage_guess` | QSTS 作为下一时刻 `PFState` 传入求解器，供自研算法 warm start；当前 OpenDSS 后端保持已编译电路状态。 |
| BESS SOC | `BESSModel` 内部状态 | 在时刻收敛后通过 `advance_state()` 更新；下一时刻限制可用充放电功率。 |

基础负荷、PV、Wind、EV 的计划 P/Q 不由上一时刻传递，而是由各自 profile 在每个时刻覆盖更新。PV Volt-VAR 的 Q 也是该时刻独立求得；BESS SOC 是当前唯一显式的跨时段设备物理状态。

## 新增算例、新 DER 与自研求解器

新增算例的基本步骤：

1. 在 `data/generate_data/` 新建生成脚本。
2. 在 `data/<case_name>/` 生成 `network.json`、`network.dss`、`load_profiles.csv` 和 `der_profiles/`。
3. 设备定义中提供 `kind`、`bus`、额定参数和 `profile_file`。
4. 使用 `python run.py --data-dir data/<case_name>` 验证。

新增 DER 类型时，继承 `DERModel`，在 `DERModel.SUPPORTED_CONTROLLERS` 登记控制器类型，并实现 profile 校验、注入、控制方程和状态更新。新增自研求解器时，在 `src/solvers/custom_pf_solver.py` 实现 `build()` 与 `solve()`，读取相同的 `NetworkModel`、`OperatingPoint` 和 `PFState`，返回 `PFResult`；QSTS、数据格式和 DER 控制器保持不变。
