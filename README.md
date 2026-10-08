# OpenDSS DER QSTS / RMS 仿真框架

本项目面向配电网中多类分布式能源（DER）的时序潮流与秒级动态仿真。网络拓扑、负荷时序、DER 时序和动态参数均由数据目录描述；QSTS 与 RMS 分别使用小时级和秒级时间尺度。

```text
DER profiles -> QSTS -> PFResult -> DynamicOperatingPoint -> RMS
```

QSTS 计算每个时刻的稳态潮流，并传递储能 SOC 等慢状态。RMS 从一个已收敛的 QSTS 工作点初始化，计算 GFL/GFM/聚合 DER 的短时动态响应。

## 目录结构

```text
data/
├── generate_data/
│   ├── ieee13_unbalanced_der.py   # IEEE 13 节点三相不平衡数据生成器
│   └── topology_plot.py           # 拓扑图绘制工具
└── ieee13_unbalanced_der/
    ├── network.json
    ├── network.dss
    ├── ieee13_base/               # IEEE 13 节点 OpenDSS 基础模型
    ├── load_profiles.csv
    ├── der_profiles/
    ├── rms_config.json
    └── network_topology.png

src/
├── power_flow.py                  # 网络、潮流输入/输出和求解器接口
├── profile_store.py               # 负荷与 DER 的 T × N 时序数组和 SOC 批量更新
├── der_model.py                   # PV、风电、储能、EV 的模型
├── qsts.py                        # QSTS 时序调度
├── rms_model.py                   # RMS 工作点、设备模型和时间推进
└── solvers/
    ├── custom_pf_solver.py         # 自研三相潮流求解器的接口骨架
    ├── fem_solver.py               # 受限线路模型的 P1 图有限元原型
    └── opendss_solver.py          # OpenDSS 潮流求解器实现

run.py                             # QSTS 运行入口
run_rms.py                         # RMS 运行入口
output/                            # 仿真结果
```

`data/<case>/network_topology.png` 展示节点、支路、负荷节点与 DER 接入位置。拓扑图是数据生成过程的一部分。

## 安装与运行

安装依赖：

```bash
pip install -r requirements.txt
```

### IEEE 13 节点三相不平衡高 DER 渗透率算例

生成数据：

```bash
python data/generate_data/ieee13_unbalanced_der.py
```

生成器使用 IEEE 13 Node Test Feeder 的 OpenDSS 基础模型；首次生成时会获取该公开基础文件。随后运行：

```bash
python run.py --data-dir data/ieee13_unbalanced_der
```

结果位于 `output/ieee13_unbalanced_der/`。该算例用于验证三相不平衡网络、单相/三相 DER 以及多设备 Volt-VAR 控制下的 QSTS 流程。

### 有限元原型与 OpenDSS 对照

```bash
python experiments/fem_radial_experiment.py --segments 60 --hours 24
python run.py --data-dir work/fem_radial_case --solver fem
```

对照结果在 `output/fem_radial_experiment/`。此原型只支持固定三相电源、串联线路、星形 PQ 负荷及 PV/风电/EV，支持三相线路互阻抗与单相接入；不支持完整 IEEE 13 节点算例中的变压器、调压器、开关、并联电容、线路充电电容等元件。数学推导、文献与本机验证见 [PDE/FEM 调研稿](docs/PDE_FEM_调研与本机实验.md)。

科研评估请优先运行基于 IEEE 13 公开线路数据的**明确删改子网**，再运行独立标注为合成数据的规模压力实验：

```bash
python experiments/ieee13_derived_benchmark.py --repeats 15
python experiments/fem_scaling_study.py --sizes 30 60 120 240 --repeats 7 --hours 24
```

前者保留选定 IEEE 线路的阻抗矩阵、相别和长度，但删去了本求解器不支持的设备，**不是完整 IEEE 13 标准算例**；DER 时序仍来自项目生成的数据。实验记录了全相节点误差、电流、损耗、控制收敛和重复计时。报告给出具体删改和方法优劣，不能将计时差解释成有限元本身的普遍加速。

## 数据接口

每个算例目录都是独立输入。程序不假定节点数量、DER 数量或网络名称。

### `network.json`

`network.json` 是 Python 侧的规范网络描述。它包含 source、母线相别与电压等级、线路代码及三相矩阵、线路/变压器/调压器/开关支路、并联电容器、基础负荷、DER 和仿真时间轴。`NetworkModel` 在构造时校验母线和设备/支路引用、相别与端子、线路矩阵维数、绕组完整性与电压等级、设备接入电压及正额定值；随后提供母线、设备、支路和并联设备的整数索引，供自研 solver 直接预处理。

```json
{
  "schema_version": "1.0",
  "base": {"frequency_hz": 60.0, "slack_bus": "SourceBus"},
  "source": {"bus": "SourceBus", "base_kv_ll": 115.0, "voltage_pu": 1.0001},
  "simulation": {
    "start_datetime": "2026-01-01 00:00:00",
    "end_datetime": "2026-01-02 00:00:00",
    "step_seconds": 3600
  },
  "buses": [{"id": "671", "phases": [1, 2, 3], "nominal_kv_ll": 4.16}],
  "line_codes": {"mtx601": {"unit": "mi", "r_ohm_per_unit": [[0.3465]], "x_ohm_per_unit": [[1.0179]]}},
  "branches": [
    {"id": "line_671_675", "kind": "line", "from_bus": "671", "to_bus": "675", "line_code": "mtx601"},
    {"id": "XFM1", "kind": "transformer", "from_bus": "633", "to_bus": "634", "windings": []}
  ],
  "shunt_devices": [{"id": "Cap1", "kind": "capacitor", "bus": "675", "q_kvar": 600.0}],
  "devices": [
    {
      "id": "PV_671_A",
      "kind": "pv",
      "bus": "671",
      "phases": [1],
      "p_rated_kw": 70.0,
      "profile_file": "der_profiles/pv_671_a.csv"
    }
  ]
}
```

时间点由 `start_datetime`、`end_datetime` 和 `step_seconds` 唯一确定，区间为 `[start_datetime, end_datetime)`。例如 24 小时、1 小时时间步会得到 24 个 QSTS 时刻。

`network.dss` 是 OpenDSS 侧的网络模型。它保存电源、线路、变压器、固定负荷和基础网络参数；DER 注入值来自各自的时序文件。

### 负荷时序：`load_profiles.csv`

负荷文件是长表；每行表示一个基础负荷在一个时刻的复功率：

```text
timestamp,load_name,p_kw,q_kvar
2026-01-01 00:00:00,Load1,33.264,10.933
2026-01-01 00:00:00,Load2,39.050,11.390
```

`load_name` 对应 `network.json` 中 `kind="load"` 的设备 ID，也是 OpenDSS 基础负荷元件名称。每个仿真时刻必须恰好包含全部基础负荷各一行；缺失、重复或额外时间戳会报错。QSTS 在每个时刻写入该时刻全部基础负荷的 `kW/kvar`。

### DER 时序：`der_profiles/<device>.csv`

每台 DER 单独使用一个文件，均含 `timestamp` 和 `controller_type`，另外按设备种类使用不同的有功/无功列：

```text
timestamp,pv_kw,pv_kvar,controller_type,q_limit_kvar,v_ref_pu,droop_kvar_per_pu
2026-01-01 12:00:00,150.0,0.0,volt_var,45.0,1.0,540.0
```

| DER 模型 | 必需有功列 | 可选无功列 | P 的符号 |
|---|---|---|---|
| `PVModel` | `pv_kw` | `pv_kvar` | 正值为发电 |
| `WindModel` | `wind_kw` | `wind_kvar` | 正值为发电 |
| `BESSModel` | `bess_kw` | `bess_kvar` | 正值放电，负值充电 |
| `EVModel` | `ev_kw` | `ev_kvar` | 正值为充电负荷 |

常用控制器类型为：

- `constant_pq`：按给定 `p_kw`、`q_kvar` 注入。
- `volt_var`：有功按时序给定，无功由电压—无功曲线计算。
- `soc_schedule`：储能按时序功率和 SOC 约束运行。

Volt-VAR 文件使用 `q_limit_kvar`、`v_ref_pu` 与 `droop_kvar_per_pu`；储能的容量、额定功率、SOC 边界和效率由 `network.json` 中对应设备定义给出。DER 模型将控制器类型与该时刻参数一起封装为潮流命令。

`phases` 表示 DER 实际接入的 OpenDSS 端子。Volt-VAR 控制只读取这些端子的电压；三相设备使用其接入相电压幅值的平均值。因此单相 DER 不会被其他相的电压直接稀释控制量。

## 算例设定

### `ieee13_unbalanced_der`

该算例基于 IEEE 13 Node Test Feeder 的三相不平衡网络。规范 JSON 包含源侧变压器、三台单相调压器及其控制参数、`633–634` 降压变压器、全部线路三相 R/X/C 矩阵、`671–692` 开关、两组固定电容器，以及各负荷的接线方式和 OpenDSS 负荷模型。DER 以单相和三相形式分散接入，形成较高的渗透率：

| 类别 | 设备 | 合计容量/规模 |
|---|---|---:|
| 光伏 | PV_632_A、PV_670_C、PV_671_A、PV_675_A、PV_675_C、PV_634_ABC | 550 kW |
| 储能 | BESS_671_B、BESS_645_B、BESS_684_A | 210 kW / 840 kWh |
| EV | EV_671_C、EV_675_B、EV_611_C | 150 kW 充电功率 |
| 风电 | Wind_680_ABC | 120 kW |

PV 采用各自的 Volt-VAR 时序控制；储能在白天充电、傍晚放电，并受额定功率、能量容量、效率和 SOC 上下限约束；EV 在晚间充电窗口接入；风电按给定有功时序运行。IEEE 基础馈线来源见 [IEEE13Nodeckt.dss](https://github.com/dss-extensions/electricdss-tst/blob/master/Version8/Distrib/IEEETestCases/13Bus/IEEE13Nodeckt.dss)。

## 潮流求解接口

`src/power_flow.py` 将网络、潮流输入、潮流结果和求解器状态分开：

| 对象 | 含义 |
|---|---|
| `NetworkModel` | 静态网络、设备定义、仿真时间轴 |
| `OperatingPoint` | 一个时刻的负荷 P/Q、DER 命令和元数据 |
| `PFState` | 由本时刻传给下一时刻的求解器状态，例如电压初值或后端状态 |
| `PFResult` | 求解器返回的收敛标志、电压、原始支路潮流、系统汇总、最终 DER 命令和下一状态 |
| `SolverContext` | 时间步长、上一时刻结果、控制迭代设置及扩展方程入口 |
| `PowerFlowSolver` | 潮流求解器协议 |
| `SecurityEvaluator`（位于 `qsts.py`） | 基于 `NetworkModel.constraints` 和 `PFResult` 统一计算电压、热限、反向潮流、VUF 与运行裕度 |

求解器接口为：

```python
solver.build(network)
result = solver.solve(
    operating_point=operating_point,
    state=previous_pf_state,
    context=solver_context,
)
```

`PowerFlowSolver` 可以承载自研牛顿法、前推回代、潮流—控制器联立方程或其他后端。它只负责形成 `PFResult` 中的电压、支路潮流、系统量和最终 DER 命令；QSTS 内部的 `SecurityEvaluator(network)` 在其后以同一套规则计算指标，因此不同 solver 的结果可直接比较。`SolverContext.extra_equations` 与 `options` 为扩展变量、约束和算法设置留出入口。QSTS 不依赖 OpenDSS 专有 API，只依赖该接口和 `PFResult` 合约。

## OpenDSS 求解过程

`OpenDSSSolver` 是 `PowerFlowSolver` 的一个实现。每个 QSTS 时刻中：

1. 写入该时刻所有基础负荷和所有 DER 命令。
2. 调用 OpenDSS 求解网络代数潮流。
3. 对 Volt-VAR DER，根据其实际端子电压更新无功命令。
4. 若命令变化超过容差，则重新写入全部负荷与全部 DER 命令并再次求解。
5. 直到 Volt-VAR 命令收敛或达到控制迭代上限。

最终潮流结果中的 `final_der_commands` 是控制迭代收敛后的命令。储能使用 OpenDSS `DispMode=EXTERNAL`，以带符号的 `kW` 指令表达充/放电：正值为向网络注入功率，负值为从网络吸收功率。SOC 更新使用最终命令和固定 QSTS 时间步长，随后传递至下一时刻。

`PowerFlowSolver.solve()` 必须为每台 DER 返回 `final_der_commands`。QSTS 会检查其设备集合与 profile 中的 DER 集合完全一致；缺失或额外设备立即报错，不会回退到原始 profile 命令。

给定负荷和 DER 注入时，网络求解的目标是三相非线性潮流方程：

\[
\frac{S_i^*}{V_i^*}=\sum_jY_{ij}V_j.
\]

对 Volt-VAR 设备，还要同时满足其 $Q=g(V)$ 控制方程。OpenDSS 后端用固定点方式形成这两个条件的共同稳态解；自研求解器可将它们直接写入同一个牛顿方程组。

## QSTS 状态传递

`run_qsts()` 的每一步遵循以下顺序：

```text
上一时刻 PFState、DER 慢状态
        ↓
读取当前负荷与每台 DER 的时序记录
        ↓
构造 OperatingPoint
        ↓
PowerFlowSolver.solve()
        ↓
PFResult.final_der_commands
        ↓
更新 BESS SOC 等 DER 慢状态，并保存 PFState
        ↓
下一时刻
```

这使得储能能量、求解器电压初值以及后端状态沿时间轴连续传递；QSTS 的第一个时刻也包含在输出中。

## QSTS 输出

`run.py` 在 `output/<case>/` 保存：

| 文件 | 内容 |
|---|---|
| `qsts_bus_voltages.csv` | 每个时刻、每个母线端子的电压幅值、相角、标幺值 |
| `qsts_branch_results.csv` | 每个时刻的线路、开关、变压器和调压器潮流；包含最大电流、额定电流、kVA 额定值、loading%、热越限、首端 P/Q、反向潮流标记，以及变压器/调压器 tap |
| `qsts_system.csv` | 系统源端功率、总负荷、DER 最终 P/Q、BESS SOC、收敛和迭代统计，以及约束与运行裕度汇总 |
| `qsts_constraints.csv` | 电压越限数量、三相电压不平衡、线路/变压器热越限数量、最大 loading% 和源侧反向潮流 |
| `qsts_hosting_capacity_metrics.csv` | 观测到的电压、线路和变压器裕度，以及最紧约束类别 |

输出目录由 `--output` 指定；未指定时为 `output/<data-dir 的末级目录>/`。

`constraints` 块定义电压上下限、电压不平衡限值、线路/变压器 loading 限值与源侧反向潮流容差。三相不平衡使用负序/正序电压幅值比：

\[
\mathrm{VUF}=100\frac{|V_2|}{|V_1|}\%.
\]

线路 loading 使用 `max_current_a / normal_amps`；变压器和调压器 loading 使用首端各相视在功率幅值之和除以首绕组 kVA 额定值。`HostingCapacityMetrics` 表示当前工作点的运行裕度，不是通过重复增大 DER 出力得到的正式 hosting-capacity 上限。IEEE13 标准基础模型不提供导线 ampacity；其中 `normal_amps` 明确标记为 `engineering_screening_assumption`，用于约束筛查，工程研究应替换为实际导线/开关额定值。

## DER 模块与控制方程

`ProfileStore` 在启动时读取并校验全部负荷与 DER profile，形成按时间、设备索引的 $T\times N$ NumPy 数组。QSTS 在每个时刻直接读取数组的一行，批量计算全部 BESS 的可用充放电功率和 SOC 更新；它不保留每台 DER 一个 DataFrame，也不在时间循环内按时间戳筛选表格。

`DERModel` 保留为单设备慢时标模型接口，可用于设备原型和扩展；标准 QSTS 路径由 `ProfileStore` 生成求解器无关的 `DERCommand`。两者使用相同的 profile 字段、控制器类型和 BESS 状态方程。

```python
commands = profiles.commands_at(time_index, dt_hours)
# 求解器返回最终控制命令后：
profiles.advance_bess(final_commands, dt_hours)
```

`ProfileStore` 将设备的 `phases` 写入 `DERCommand.parameters["terminal_nodes"]`，使求解器知道控制器应使用哪些端子电压。代数控制律由 `PowerFlowSolver` 处理。

### PV Volt-VAR

当 `controller_type=volt_var` 时，有功仍取 profile 的 `pv_kw`，无功根据实际端子电压幅值的平均标幺值 $V$ 更新：

\[
Q_{cmd}=\operatorname{clip}\left[k\left(V_{ref}-V\right),-Q_{max},Q_{max}\right],
\]

其中 $k$ 是 `droop_kvar_per_pu`，$V_{ref}$ 是 `v_ref_pu`，$Q_{max}$ 是 `q_limit_kvar`。低电压时得到正无功命令，高电压时得到负无功命令。对于单相 DER，$V$ 仅来自其所接入相；三相 DER 对所接入各相求平均。

该控制律是代数、无状态的。在 OpenDSS 后端中，它通过“潮流 → 由电压更新 Q → 潮流”的内层固定点迭代与网络方程共同收敛，而非由 QSTS 在求解器外重复调用。

### BESS SOC

`BESSModel` 是当前唯一具有显式慢时标物理状态的 DER。求解前，计划功率首先受额定功率、可放电能量和可充电余量限制；求解收敛后，使用最终的有功命令更新 SOC。令 $E$ 为容量（kWh），$\eta_{dis}$、$\eta_{ch}$ 为放电和充电效率，正功率表示放电：

\[
SOC_{t+1}=SOC_t-
\frac{P_t\Delta t}{\eta_{dis}E}\times100,\qquad P_t>0,
\]

\[
SOC_{t+1}=SOC_t+
\frac{|P_t|\eta_{ch}\Delta t}{E}\times100,\qquad P_t<0.
\]

更新结果限制在 `[reserve_soc, 100]`。`qsts_system.csv` 中 `<device_id 小写>_soc_pct` 表示本时刻结束后、传递到下一时刻的 SOC。

### OpenDSS 元件映射与符号

`OpenDSSSolver` 根据 `kind` 写入对应 OpenDSS 元件：PV 与风电使用 `Generator`，BESS 使用 `Storage`，EV 使用 `Load`。发电型 DER 的正有功向网络注入；EV 的正 `ev_kw` 作为负荷从网络吸收功率。BESS 采用 `DispMode=EXTERNAL`，并直接写入带符号的 `kW`：正值放电、负值充电。控制迭代中每次潮流前都会重新写入所有 DER 命令，因此储能不会在同一 QSTS 时刻的 Volt-VAR 内层迭代中重复推进。

## RMS 仿真

RMS 以 QSTS 某一时刻的已收敛断面为初值。`DynamicOperatingPoint` 保存母线电压、负荷 P/Q、DER P/Q、DER 慢状态及时间戳；`RMSState` 保存秒级时间、设备动态状态、电压和潮流状态。

`rms_config.json` 只描述 QSTS 数据中没有的动态信息，例如设备动态模型、PLL 参数、内外环时间常数、下垂系数、虚拟惯量、电流限值、扰动和仿真步长。网络拓扑和 QSTS 时序数据不在该文件重复出现。

已实现的设备模型：

- `GFLInverter`：同步旋转坐标系 PLL、P/Q 外环、电流内环和圆形限流。
- `GFMInverter`：P-f、Q-V 下垂、虚拟惯量/阻尼和内部电压动态。
- `AggregateDER`：聚合 DER 的 P/Q 一阶动态与电压支撑。

RMS 每个步长先根据设备状态和参考值计算 DER 注入，再调用 `PowerFlowSolver` 解网络代数方程，之后推进设备微分状态。它适合控制器和配电网机电暂态级别的原型验证，不等同于电磁暂态（EMT）仿真。

### 初始化、状态与数值推进

`load_qsts_operating_point()` 从 `network.json`、`load_profiles.csv`、`qsts_bus_voltages.csv` 和 `qsts_system.csv` 重建指定时刻的 `DynamicOperatingPoint`。其中包含各相复电压、全部基础负荷 P/Q、QSTS 最终 DER 命令以及 BESS SOC。网络拓扑不在 RMS 输出中重复保存。

`initialize_rms()` 对每一台配置为动态模型的设备，以 QSTS 终值初始化内部状态，满足：

\[
P(0)=P^\star,\qquad Q(0)=Q^\star,\qquad V(0)=V^\star.
\]

`RMSState` 保存秒级时间、各设备微分状态、各母线相电压和 `PFState`。三相母线用正序分量计算 RMS 控制器使用的电压幅值与相角；单相母线直接使用其现有相电压。

对步长 \(\Delta t\)，`run_rms()` 的顺序为：

```text
QSTS 工作点 -> RMS 初始化 -> t = 0 的网络潮流
每个 RMS 步长：
    应用到时的负荷阶跃或 DER 跳闸事件
    用上一网络电压计算各 RMSDevice 的状态导数
    显式 Euler 推进设备状态
    从新状态生成 DER P/Q 命令
    PowerFlowSolver.solve() 重解网络代数方程
    保存状态、母线电压和汇总量
```

这是“DER 微分状态 + 准稳态网络代数潮流”的 DAE 原型：设备状态为 \(x\)，母线电压为代数变量 \(z\)。网络仍由 `PowerFlowSolver` 提供，因此 RMS 编排不依赖 OpenDSS 的具体类，也可使用后续的自研求解器。

### 动态 DER 模型

各模型都使用设备 `p_rated_kw` 作为 P/Q 标幺基值，并使用 `current_limit_pu` 执行圆形限流：

\[
\frac{\sqrt{P^2+Q^2}}{\max(V,\epsilon)}\le I_{max}.
\]

超过限值时，P/Q 或 d/q 电流分量按同一比例缩小，以保持功率因数角。

`aggregate_der` 采用一阶 P/Q 跟踪和电压无功支撑：

\[
q_{ref}\leftarrow q_{ref}+k_v(V_{ref}-V),\qquad
\tau_p\dot p_{out}=p_{ref}-p_{out},\quad
\tau_q\dot q_{out}=q_{ref}-q_{out}.
\]

它还包含 `p_measure_pu`、`q_measure_pu` 的一阶测量滤波状态。

`gfl_inverter` 包含同步旋转坐标系 PLL、P/Q 外环和 d/q 电流内环。PLL 使用

\[
e_{pll}=\sin(\theta_{bus}-\theta_{pll}),\qquad
\Delta f=k_p e_{pll}+k_i\int e_{pll}\,dt,
\]

并以 \(\dot\theta_{pll}=2\pi\Delta f\) 推进相角。其状态包括 P/Q 滤波量、d/q 电流、PLL 相角和积分器；`metrics()` 输出估计频率。

`gfm_inverter` 包含虚拟惯量、阻尼、P-f/Q-V 下垂和内部电压环：

\[
p_{cmd}=p_{ref}-k_{pf}\Delta f,\qquad
q_{cmd}=q_{ref}+k_{qv}(V_{ref}-V),
\]

\[
2H\dot{\Delta f}=p_{ref}-p_{meas}-D\Delta f.
\]

它输出频率和内部电压指标，并以一阶功率、测量和电压时间常数表征控制动态。

### `rms_config.json`

该文件只描述 QSTS 时序中没有的动态模型和场景。每个 `device_id` 必须引用 `network.json` 内已有的非负荷设备，不重复定义拓扑、接入母线或额定功率。IEEE-13 算例的配置结构为：

```json
{
  "nominal_frequency_hz": 60.0,
  "dynamic_devices": [
    {
      "device_id": "PV_671_A",
      "model_type": "gfl_inverter",
      "current_limit_pu": 1.25,
      "pll_kp_hz_per_rad": 12.0,
      "pll_ki_hz_per_rad_s": 180.0,
      "tau_p_control_s": 0.08,
      "tau_q_control_s": 0.06,
      "tau_current_s": 0.015
    }
  ],
  "scenarios": {
    "load_step": {
      "dt_s": 0.02,
      "t_end_s": 2.0,
      "events": [
        {"time_s": 1.0, "type": "load_scale", "device_id": "671", "p_multiplier": 1.1, "q_multiplier": 1.1}
      ]
    }
  }
}
```

`gfl_inverter` 需要 PLL 参数和 P/Q、电流环时间常数；`gfm_inverter` 需要惯量、阻尼、P-f/Q-V 下垂以及功率、测量和电压环时间常数；`aggregate_der` 需要 P/Q 控制和测量时间常数，可选 `voltage_support_pu_per_pu`。三类模型均要求正的 `current_limit_pu`。

事件类型如下：

- `load_scale`：在指定时刻将目标基础负荷 P/Q 分别乘以 `p_multiplier`、`q_multiplier`。
- `der_trip`：在指定时刻将目标动态 DER 的 P/Q 参考置为零，实际注入按照模型时间常数衰减。

`ieee13_unbalanced_der/rms_config.json` 包含以下场景：

- `flat_run`：无扰动运行，用于检查初始化稳态。
- `load_step`：指定时刻对一个负荷施加功率阶跃。
- `pv_trip`：在 1 s 时使 `PV_671_A` 的有功和无功参考降为零。

其中 `PV_671_A` 使用 GFL inverter，`BESS_671_B` 使用 GFM inverter，三相 `Wind_680_ABC` 使用 aggregate DER；因此同一 RMS 断面同时覆盖三类动态设备模型。

先运行 QSTS，再从同一个已保存时间戳启动 RMS：

```bash
python run.py --data-dir data/ieee13_unbalanced_der
python run_rms.py --data-dir data/ieee13_unbalanced_der --qsts-output output/ieee13_unbalanced_der --time "2026-01-01 12:00:00" --scenario flat_run
python run_rms.py --data-dir data/ieee13_unbalanced_der --qsts-output output/ieee13_unbalanced_der --time "2026-01-01 12:00:00" --scenario load_step
python run_rms.py --data-dir data/ieee13_unbalanced_der --qsts-output output/ieee13_unbalanced_der --time "2026-01-01 12:00:00" --scenario pv_trip
```

RMS 输出保存在 `output/<case>/rms/<scenario>/`；可通过 `run_rms.py --output <目录>` 覆盖：

| 文件 | 内容 |
|---|---|
| `rms_summary.csv` | 时间、设备动态状态、DER P/Q、频率及场景量 |
| `rms_bus_voltages.csv` | 每个 RMS 步长的母线电压 |
| `rms_initialization.json` | QSTS 工作点和 RMS 初值摘要 |

## 自研求解器接入

自研求解器可放在 `src/solvers/`，实现 `PowerFlowSolver` 的 `build()` 与 `solve()`。`custom_pf_solver.py` 给出了该协议的骨架；它尚不包含求解算法。至少应返回：

```python
PFResult(
    converged=...,
    bus_voltages=...,
    bus_voltage_records=...,
    branch_records=...,
    summary=...,
    next_state=...,
    final_der_commands=...,
)
```

若求解器将潮流方程与 DER 控制方程联立，`final_der_commands` 应保存联立解中的最终 DER P/Q，`next_state` 应保存下一 QSTS 时刻需要的内部状态。这样 QSTS 与 RMS 的上层流程无需针对求解器类型分支处理。

### 自研求解器的实现边界

现有接口适合自研三相潮流求解器。`build(network)` 可将 `NetworkModel.bus_index`、`device_index`、`branch_index`、`shunt_index`、`source`、`line_codes`、`buses`、`branches` 和 `shunt_devices` 转换为母线相别索引、支路索引、Ybus、雅可比稀疏结构或前推回代拓扑；这些对象在整段 QSTS 中不变。`solve()` 接收该时刻的负荷、DER 命令和上一时刻 `PFState`，并返回下一时刻可复用的状态。

若采用潮流—控制器联立牛顿法，未知量可包括母线复电压和受控 DER 的 Q（或 d/q 电流、内部控制状态）；残差由网络功率平衡、Volt-VAR 方程、下垂方程、限值互补条件等组成。联立解应写入 `PFResult.final_der_commands`，使 BESS SOC 和 RMS 初始化使用实际的最终 P/Q，而不是 profile 的原始计划值。

求解器需要自行处理以下后端相关内容：三相/单相端子连接、相序与标幺基值、负荷和 DER 的功率符号、变压器与调压器模型、收敛判据，以及 `PFState.backend_state` 中的 warm start 或内部变量。QSTS 不依赖 OpenDSS API；唯一要求是遵守 `PowerFlowSolver` 与 `PFResult` 合约。

## 组织新算例与设备模型

一个可运行算例应具备以下文件：

```text
data/<case>/
├── network.json
├── network.dss
├── load_profiles.csv
├── der_profiles/
└── network_topology.png
```

生成脚本通常位于 `data/generate_data/`，并同时生成 `network.json`、`network.dss`、所有时序 CSV 和拓扑图。`network.json` 的 `devices` 是 Python 模型和 OpenDSS 后端之间的设备索引：基础负荷使用 `kind: "load"`，DER 使用 `pv`、`wind`、`bess` 或 `ev`，每个 DER 通过 `profile_file` 指向自己的时序文件。

新 DER 慢时标模型可继承 `DERModel`，实现 profile 校验、`get_injection()`，并在有跨时段物理量时实现 `advance_state()` 与 `state_summary()`。标准批量 QSTS 路径还需要在 `profile_store.py` 的 `_POWER_COLUMNS` 登记该 `kind` 对应的 P/Q 列名，并在 `ProfileStore.from_case()` 提供该设备的向量状态、约束和状态推进规则。新控制器类型需要同时登记 `DERModel.SUPPORTED_CONTROLLERS` 与 `profile_store.py` 的控制器集合，并由某个 `PowerFlowSolver` 对 `DERCommand.controller_type` 和 `parameters` 给出代数求解规则。

RMS 动态设备模型继承 `RMSDevice`，实现 `initialize()`、`derivative()` 和 `injection()`，并登记到 `MODEL_TYPES`。`initialize()` 必须以 QSTS 的最终 P/Q 与母线电压构造稳态状态；`injection()` 返回求解器无关的 `DERCommand`；`derivative()` 不应依赖 OpenDSS 对象。这样慢时标 QSTS 模型、潮流求解器和秒级动态模型保持独立：同一个 QSTS 工作点可以对应不同的 RMS 设备配置与扰动场景。

批量 QSTS 对于大规模同类设备尤其合适。设备种类很少、但每类设备参数不同的情形可直接使用数组参数；设备具有不同离散状态机、复杂事件或强异构内部状态时，应按设备种类维护独立的批量状态数组，而不是退回到“每台设备一个 DataFrame”的模式。
