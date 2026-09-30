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
├── network_topology.png    # 由生成器绘制的拓扑图：节点、DER 与支路信息
├── load_profiles.csv       # 逐负荷 P/Q 时序
├── der_profiles/           # 每台 DER 一份时序和控制器配置
└── rms_config.json         # 可选；仅包含 RMS 动态参数和扰动场景
```

`network.json` 是固定网络的规范输入，包含母线、支路、三相阻抗矩阵、设备类型、接入节点、额定参数和 DER profile 相对路径。`network.dss` 是 OpenDSS 后端的兼容文件，由同一个算例生成器创建。

每次运行生成器还会生成 `network_topology.png`。图中方形黄色节点为平衡源，蓝色圆形节点为带基础负荷的节点；彩色菱形分别表示 PV、风机、BESS 和 EV，支路旁标注支路 ID 与长度。图仅由 `network.json` 的内容绘制，可用于检查生成数据的拓扑和设备接入是否符合预期。

## 生成和运行

安装依赖：

```powershell
pip install -r requirements.txt
```

生成并运行多支路算例：

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

负荷 profile 是长表。每一个时刻必须包含 `network.json` 中所有 `kind="load"` 设备各一行，时间用 `timestamp` 表示真实时间（`YYYY-MM-DD HH:MM:SS`）：

```csv
timestamp,load_name,p_kw,q_kvar
2026-01-01 00:00:00,Load1,270.630,90.210
2026-01-01 00:00:00,Load2,217.124,70.246
```

每台 DER 的 `profile_file` 由 `network.json` 内设备定义指定；因此 `run.py` 会自动发现并创建 PV、Wind、BESS、EV 模型，不再写死 `PV1`、`bus1` 或三个负荷。

仿真的开始时间、结束时间和时间间隔记录在 `network.json` 顶层的 `simulation` 块中：

```json
"simulation": {
  "start_datetime": "2026-01-01 00:00:00",
  "end_datetime": "2026-01-02 00:00:00",
  "step_seconds": 3600
}
```

所有 DER profile 必须包含 `timestamp` 和 `controller_type`，且其 `timestamp` 与负荷 profile 对齐。当前支持：

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
| `DERCommand` | 单台 DER 的 P/Q、控制器类型与代数控制参数（Vref、droop、Qmin/Qmax、SOC 等） |
| `PFState` | 上一时刻复电压初值与求解器状态 |
| `PFResult` | 电压、线路结果、系统汇总、迭代次数、最终 DER 命令、控制迭代次数、下一状态 |
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

对每个时间戳执行：

```text
读取当前逐负荷 P/Q 与各 DER profile
→ 构造 OperatingPoint（含 DER 控制参数）
→ PowerFlowSolver.solve(OperatingPoint, PFState)   // 解析网络方程与 DER 控制代数方程
→ BESS 推进 SOC，PFResult.next_state 传给下一时刻
```

DER 控制迭代（例如 Volt-VAR 的 `Q = g(V)`）发生在求解器内部：`OpenDSSSolver` 通过“潮流 → 更新 Q → 再潮流”的内层 fixed-point 循环使其与网络方程共同收敛，最终 DER 命令保存在 `PFResult.final_der_commands`，收敛所用的控制更新次数记录在 `PFResult.control_iterations`。QSTS 每个时间点只调用一次 `solver.solve()`，自身不再做外层控制循环。

PV Volt-VAR 使用：

```text
Q = clip[k × (Vref − Vbus), −Qmax, Qmax]
```

BESS 的 SOC 为跨时段显式状态；其容量、初始 SOC、SOC 下限和效率从 `network.json` 的 BESS 设备定义读取。PV、Wind、EV 是无状态 profile 回放模型。`PFState` 已传递给下一时刻，为自研求解器提供电压 warm start 接口。

## 输出

每个算例输出目录包含：

- `qsts_bus_voltages.csv`：每时刻、每母线相别的电压幅值和相角；
- `qsts_system.csv`：总负荷、内层 DER 控制迭代次数、潮流内部迭代次数、电压范围、损耗、源端功率、最终 DER P/Q 和 BESS SOC；
- `qsts_branch_results.csv`：每时刻、每条支路的电流幅值和首端 P/Q（由求解器的 `branch_records` 保存）。

## DER 模块详解

### 统一 DER 接口

所有 DER 均继承 `src/der_model.py` 中的 `DERModel`。QSTS 不针对 PV、风机、储能或 EV 写类型判断；它只在每个时刻向模型请求注入命令（含控制器参数），并在收敛后推进跨时段状态。代数控制律（如 Volt-VAR）由求解器读取 `DERCommand` 中的 `controller_type` 与参数自行求解，不再由 QSTS 回调节点。

```python
class DERModel:
    def _validate_profile(self) -> None:
        """校验设备时序和 controller_type。"""

    def get_injection(self, time_step: datetime, dt_hours: float) -> dict[str, object]:
        """按时间戳返回计划 P/Q、控制器类型和参数。"""

    def to_command(self, injection: dict[str, object]) -> DERCommand:
        """将模型数据转为求解器无关的 DERCommand（含控制类型和参数）。"""

    def advance_state(self, injection: dict[str, object], dt_hours: float) -> None:
        """将本时刻状态传递到下一时刻。"""
```

`get_injection()` 和 `to_command()` 只产出当前时刻的计划 P/Q 与控制参数，不计算控制响应；VOLT-VAR 这类代数控制律由求解器依据 `DERCommand` 求解。`advance_state()` 只在当前时刻收敛后调用，用于跨时段物理状态（如 BESS SOC）。

### 当前 DER 类型

| 模型 | P 的符号 | 控制类型 | 跨时段状态 |
| --- | --- | --- | --- |
| `PVModel` | 正值为发电 | `volt_var` 或 `constant_pq` | 无 |
| `WindModel` | 正值为发电 | `constant_pq` | 无 |
| `BESSModel` | 正值放电、负值充电 | `soc_schedule` | SOC |
| `EVModel` | 正值为充电负荷 | `constant_pq` | 无 |

DER 模型本身与求解器后端解耦，不包含 OpenDSS 元件概念；`kind`（`pv`/`wind`/`bess`/`ev`）映射到哪种 OpenDSS 元件（`Generator`/`Storage`/`Load`）由 `OpenDSSSolver._set_der_command()` 决定。数据流统一为：

```text
DER 数据/状态/控制 → DERCommand → PowerFlowSolver → OpenDSS / 自研求解器
```

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
timestamp,pv_kw,pv_kvar,controller_type,q_limit_kvar,v_ref_pu,droop_kvar_per_pu
2026-01-01 12:00:00,180,0,volt_var,45,1.0,540
```

BESS 的计划功率 profile：

```csv
timestamp,bess_kw,bess_kvar,controller_type
2026-01-01 10:00:00,-45,0,soc_schedule
2026-01-01 18:00:00,35,0,soc_schedule
```

`controller_type` 必须逐时刻给出，因此后续可在同一台设备的不同运行时段切换控制方式。当前实现支持 `constant_pq`、`volt_var` 和 `soc_schedule`；不支持的类型会在 profile 读取时明确报错。

### PV Volt-VAR 方程

PV 在潮流求解后读取其接入母线所有相别的平均标幺电压 \(V_{bus}\)，并计算：

\[
Q_{PV}=\operatorname{clip}\left[k(V_{ref}-V_{bus}),-Q_{max},Q_{max}\right]
\]

低电压时注入正无功，高电压时吸收无功。该控制器是代数、无状态的：每个时刻电压确定后，Q 由该式唯一确定。代数求解由 `PowerFlowSolver` 内部完成——`OpenDSSSolver` 以“潮流 → 更新 Q → 再潮流”的 fixed-point 循环使网络方程与该式共同收敛，最终收敛的 Q 写入 `PFResult.final_der_commands`。

### BESS SOC 状态方程

BESS 在写入求解器前先根据当前 SOC 约束其计划功率，保证不会突破最小 SOC 或 100% SOC。收敛后，以最终实际功率更新：

\[
SOC_{t+1}=SOC_t-\frac{P_t\Delta t}{\eta_{dis}E}\times100,\quad P_t>0
\]

\[
SOC_{t+1}=SOC_t+\frac{|P_t|\eta_{ch}\Delta t}{E}\times100,\quad P_t<0
\]

其中 \(E\)、SOC 下限、初始 SOC 和效率从该 BESS 的网络设备定义读取；\(\Delta t\) 为该时间点到下一时间点的时间间隔（小时），由相邻 `timestamp` 之差求得，因此 BESS 的能量约束不依赖固定步长。`qsts_system.csv` 的 `<bess_name>_soc_pct` 是每个时刻末、将传给下一时刻的 SOC。

## QSTS 求解过程详解

对每一个 `timestamp`，QSTS 的实际顺序为：

```text
1. 从 load_profiles.csv 读取所有基础负荷的 P/Q
2. 从各 DER profile 读取计划 P/Q、controller_type 与控制参数
3. 构造 OperatingPoint（含 DERCommand 控制律），并携带上一时刻 PFState 调用求解器
4. 求解器内部联合求解网络方程与 DER 控制代数方程，直到共同收敛
5. 获取 PFResult：电压、线路结果、系统汇总、最终 DER 命令、控制迭代次数
6. 推进 BESS 等跨时段状态，PFState 传给下一时刻
7. 保存电压、最终 DER 命令、SOC、损耗和迭代信息
```

对于网络方程，求解器在给定负荷和 DER 注入时求解三相非线性潮流：

\[
\frac{S_i^*}{V_i^*}=\sum_jY_{ij}V_j
\]

当前 `OpenDSSSolver` 用 OpenDSS 完成该方程组求解，并以“潮流 → 按 `DERCommand` 控制律更新 Q → 再潮流”的内层固定点迭代把网络方程与 DER 控制方程耦合到同一稳态解；`control_iterations` 记录该内层控制更新次数，`pf_iterations` 记录 OpenDSS 单次潮流的内部迭代次数，最终收敛命令存于 `PFResult.final_der_commands`。

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

## QSTS 到 RMS 动态仿真

RMS 首版建立在既有 QSTS 结果之上：先用小时级 QSTS 找到已收敛的运行断面，再从指定时刻切换到秒级动态仿真。

```text
DER profiles → QSTS → saved PFResult-equivalent data → RMS initialization → RMS simulation
```

`DynamicOperatingPoint` 是已求解的运行断面，在 [rms_model.py](src/rms_model.py) 中定义，包含指定 QSTS 时刻的各相复电压、每个基础负荷的 P/Q、每台 DER 的最终 P/Q、已保存的慢状态（当前为 BESS SOC）以及 QSTS 汇总元数据。它由以下已有文件重建，不复制网络拓扑或时序数据：

- `network.json`
- `load_profiles.csv`
- `output/<case>/qsts_bus_voltages.csv`
- `output/<case>/qsts_system.csv`

### RMS 模型与边界

当前提供三种标幺化的动态设备模型，共用统一 `RMSDevice` 接口：`initialize()`（以 QSTS 断面初始化状态）、`derivative()`（返回状态方程右端）、`injection()`（由状态生成注入网络的 P/Q 命令）与 `metrics()`（输出观测指标）。三种模型仅通过“标幺目标功率 → 内部状态方程 → 注入网络的 P/Q 命令”交互，不修改网络拓扑；设备额定功率 `rating_kw` 取自 `network.json` 的 `p_rated_kw`，标称频率取自 `rms_config.json` 的 `nominal_frequency_hz`（缺失时回退 `network.json` 的 `base.frequency_hz`，默认 60 Hz）。

| 模型 | 类型 | 关键状态 |
| --- | --- | --- |
| `aggregate_der` | 聚合 DER（一阶 P/Q 跟踪 + 无功电压支撑） | \(p_{out},q_{out},p_{meas},q_{meas},V_{ref}\) |
| `gfl_inverter` | 跟网型逆变器（GFL） | \(p_f,q_f,i_d,i_q,\theta_{pll},\int\varepsilon\) |
| `gfm_inverter` | 构网型逆变器（GFM） | \(p_{out},q_{out},p_{meas},q_{meas},\Delta f,\theta,V_{int},V_{ref}\) |

所有功率均为以各自设备 `p_rated_kw` 为基值的标幺值；\(V_{bus}\) 为设备接入母线的正序电压幅值，\(\theta_{bus}\) 为正序电压相角（由三相相电压经对称分量法提取，见 `_bus_voltage()`）。每个模型都带一个圆形电流限幅，由参数 `current_limit_pu` 给出：当视在电流 \(|S|/V_{bus}\) 超过限值时，按比例缩减 P、Q（见 `_clip_current()`）。

#### `aggregate_der`（聚合 DER）

一阶 P/Q 跟踪、无功电压支撑和测量低通。记标幺目标为 \(p_{ref},q_{ref}\)：

\[
\begin{aligned}
q_{ref} &\leftarrow q_{ref}+k_{v}(V_{ref}-V_{bus}), \\
(p_{ref},q_{ref}) &\leftarrow \mathrm{clip}(p_{ref},q_{ref}), \\
\tau_p \dot p_{out}&=p_{ref}-p_{out},\qquad \tau_q \dot q_{out}=q_{ref}-q_{out}, \\
\tau_m \dot p_{meas}&=p_{out}-p_{meas},\qquad \tau_m \dot q_{meas}=q_{out}-q_{meas}.
\end{aligned}
\]

其中 \(k_{v}\) 即 `voltage_support_pu_per_pu`（可选，默认 0），\(\mathrm{clip}\) 为上述圆形电流限幅。注入命令由 \(p_{out},q_{out}\) 再经一次限幅后乘以 \(p_{rated}\) 得到。

#### `gfl_inverter`（跟网型逆变器）

包含 SRF-PLL、P/Q 外环、电流内环与限流。记 PLL 相角误差 \(\varepsilon=\sin(\theta_{bus}-\theta_{pll})\)，频率偏差 \(\Delta f=k_p^{pll}\varepsilon+k_i^{pll}\int\varepsilon\)：

\[
\begin{aligned}
\tau_p \dot p_f &= p_{ref}-p_f,\qquad \tau_q \dot q_f=q_{ref}-q_f, \\
(i_{d,ref},i_{q,ref}) &= \mathrm{clip}\bigl(p_f/V_{bus},\,q_f/V_{bus}\bigr), \\
\tau_i \dot i_d &= i_{d,ref}-i_d,\qquad \tau_i \dot i_q=i_{q,ref}-i_q, \\
\dot\theta_{pll} &= 2\pi\,\Delta f,\qquad \frac{d}{dt}\int\varepsilon=\varepsilon.
\end{aligned}
\]

注入命令由 \(i_d,i_q\) 经电压还原为功率并限幅后乘以 \(p_{rated}\)。`metrics()` 输出 \(f_{nom}+\Delta f\)。

#### `gfm_inverter`（构网型逆变器）

包含虚拟惯量摆动方程、P-f/Q-V 下垂和电压内环。记频率偏差 \(\Delta f\)，下垂命令：

\[
\begin{aligned}
p_{cmd}&=p_{ref}-k_{pf}\Delta f,\qquad q_{cmd}=q_{ref}+k_{qv}(V_{ref}-V_{bus}), \\
(p_{cmd},q_{cmd}) &\leftarrow \mathrm{clip}(p_{cmd},q_{cmd}), \\
2H\,\dot{\Delta f} &= p_{ref}-p_{meas}-D\,\Delta f,\qquad \dot\theta=2\pi\,\Delta f, \\
\tau_p \dot p_{out} &= p_{cmd}-p_{out},\qquad \tau_p \dot q_{out}=q_{cmd}-q_{out}, \\
\tau_m \dot p_{meas} &= p_{out}-p_{meas},\qquad \tau_m \dot q_{meas}=q_{out}-q_{meas}, \\
V_{tgt}&=V_{ref}+k_{vq}(q_{ref}-q_{meas}),\qquad \tau_v \dot V_{int}=V_{tgt}-V_{int}.
\end{aligned}
\]

其中 \(k_{pf}\)、\(k_{qv}\)、\(H\)、\(D\)、\(k_{vq}\) 分别对应 `p_droop_pu_per_hz`、`q_droop_pu_per_pu`、`inertia_s`、`damping_pu_per_hz`、`voltage_droop_pu_per_pu`（可选，默认 0）。注入命令由 \(p_{out},q_{out}\) 限流后乘以 \(p_{rated}\)。`metrics()` 输出 \(f_{nom}+\Delta f\) 与 \(V_{int}\)。

#### 数值推进与网络求解

初始化严格采用 QSTS 断面：\(P(0)=P^\star\)、\(Q(0)=Q^\star\)、\(V(0)=V^\star\)。每个秒级步长先按显式 Euler 推进各设备的内部状态，再由 `injection()` 得到其注入网络的 P/Q，随后通过 `PowerFlowSolver.solve()` 重解网络代数方程，用得到的母线电压推进下一步。因此当前版本是“动态设备状态 + 准稳态网络代数求解”的 RMS 验证框架，不是电磁暂态仿真。后续可保留 `RMSDevice` 接口，替换为更完整的 GFL/GFM/DER_A 或同步机模型，并替换网络代数求解部分。

`RMSState` 保存当前秒级时间、设备动态状态 \(x\)、母线复电压 \(z\) 及潮流 warm-start 状态；`RMSResult` 返回时序汇总表、母线相电压表和最终状态。RMS 编排不依赖 `OpenDSSSolver` 的具体类、只依赖 `PowerFlowSolver`，当前入口选择 OpenDSS 后端。

### `rms_config.json`

该文件只补充 QSTS 中不存在的动态信息。`device_id` 必须引用既有 `network.json` 中的 DER，不能重复定义接入母线、额定功率或 QSTS 时序；`model_type` 必须是上述三种之一，且每台动态设备还需给出 `current_limit_pu`。例如 12 节点算例为全部五台 DER 配置了动态参数和三个场景：

```json
{
  "nominal_frequency_hz": 60.0,
  "dynamic_devices": [
    {"device_id": "PV1", "model_type": "gfl_inverter", "current_limit_pu": 1.25, "pll_kp_hz_per_rad": 12.0, "pll_ki_hz_per_rad_s": 180.0, "tau_p_control_s": 0.08, "tau_q_control_s": 0.06, "tau_current_s": 0.015},
    {"device_id": "PV2", "model_type": "gfm_inverter", "current_limit_pu": 1.25, "inertia_s": 1.5, "damping_pu_per_hz": 0.25, "p_droop_pu_per_hz": 0.08, "q_droop_pu_per_pu": 2.0, "voltage_droop_pu_per_pu": 0.5, "tau_power_control_s": 0.08, "tau_power_measure_s": 0.04, "tau_voltage_control_s": 0.05},
    {"device_id": "Wind1", "model_type": "aggregate_der", "current_limit_pu": 1.20, "tau_p_control_s": 0.25, "tau_q_control_s": 0.15, "tau_measure_s": 0.05, "voltage_support_pu_per_pu": 1.0},
    {"device_id": "BESS1", "model_type": "gfl_inverter", "current_limit_pu": 1.30, "pll_kp_hz_per_rad": 10.0, "pll_ki_hz_per_rad_s": 150.0, "tau_p_control_s": 0.10, "tau_q_control_s": 0.08, "tau_current_s": 0.02},
    {"device_id": "EV1", "model_type": "aggregate_der", "current_limit_pu": 1.10, "tau_p_control_s": 0.30, "tau_q_control_s": 0.20, "tau_measure_s": 0.10, "voltage_support_pu_per_pu": 0.0}
  ],
  "scenarios": {
    "flat_run": {"dt_s": 0.02, "t_end_s": 2.0, "events": []},
    "load_step": {
      "dt_s": 0.02,
      "t_end_s": 2.0,
      "events": [{"time_s": 1.0, "type": "load_scale", "device_id": "Load12", "p_multiplier": 1.10, "q_multiplier": 1.10}]
    },
    "der_trip": {
      "dt_s": 0.02,
      "t_end_s": 2.0,
      "events": [{"time_s": 1.0, "type": "der_trip", "device_id": "PV1"}]
    }
  }
}
```

`load_scale` 在指定时刻立即把该基础负荷的 P/Q 乘以 `p_multiplier`、`q_multiplier`；`der_trip` 在指定时刻调用该动态 DER 的 `trip()`，把其 \(P^{ref},Q^{ref}\) 目标置零，实际注入功率随后按其时间常数衰减到零。

### 运行 RMS

先生成 QSTS 输出，再用 `--time` 选定一个时间戳作为动态初始工作点：

```powershell
python run.py --data-dir data\radial_12bus
python run_rms.py --data-dir data\radial_12bus --qsts-output output\radial_12bus --time "2026-01-01 12:00:00" --scenario flat_run
python run_rms.py --data-dir data\radial_12bus --qsts-output output\radial_12bus --time "2026-01-01 12:00:00" --scenario load_step
python run_rms.py --data-dir data\radial_12bus --qsts-output output\radial_12bus --time "2026-01-01 12:00:00" --scenario der_trip
```

默认输出目录是 `output/<case>/rms/<scenario>/`：

- `rms_summary.csv`：每个秒级时刻的扰动标签、潮流内部迭代次数、最小/最大电压、网损、源侧 P/Q，以及每台动态 DER 的实际注入 P/Q 与其内部状态/指标（列名形如 `<device_id 小写>_p_kw`、`<device_id 小写>_p_filter_pu`、`<device_id 小写>_frequency_hz` 等）；
- `rms_bus_voltages.csv`：每时刻、每母线相的电压幅值和相角；
- `rms_initialization.json`：选用的 QSTS 时间戳、初始 DER P/Q 和慢状态，便于复现工作点。

新增 DER 类型时，继承 `DERModel`，在 `DERModel.SUPPORTED_CONTROLLERS` 登记控制器类型，并实现 profile 校验、注入、控制方程和状态更新。新增自研求解器时，在 `src/solvers/custom_pf_solver.py` 实现 `build()` 与 `solve()`，读取相同的 `NetworkModel`、`OperatingPoint` 和 `PFState`，返回 `PFResult`；QSTS、数据格式和 DER 控制器保持不变。
