# OpenDSS QSTS Multi-DER Demo

Python 与 DSS-Python/OpenDSS 构建的 24 小时 QSTS 演示。场景包含三段三相径向馈线、三个逐节点 P/Q 负荷，以及 PV、风电、储能、EV。输入数据和仿真分为独立两步。

## 运行

```powershell
pip install -r requirements.txt
python generate_data.py
python run.py
```

`generate_data.py` 生成并覆盖 `data/`；`run.py` 只读取这些数据，执行 `hour=0` 到 `hour=23` 的 QSTS。可使用独立目录：

```powershell
python generate_data.py --data-dir data\case_a
python run.py --data-dir data\case_a --output output\case_a
```

## 数据

```text
data/
├── feeder.dss
├── load_profiles.csv
└── der_profiles/
    ├── pv1.csv
    ├── wind1.csv
    ├── bess1.csv
    └── ev1.csv
```

`feeder.dss` 只定义静态网络与设备：12.47 kV 源端、`source → bus1 → bus2 → bus3` 三段三相线路、各元件的接入位置/相别/额定参数。基础负荷的初始 P/Q 为零，所有运行 P/Q 均来自时序文件。

`load_profiles.csv` 是逐负荷长表；每一时刻必须有 `Load1`、`Load2`、`Load3` 各一行：

```csv
hour,load_name,p_kw,q_kvar
0,Load1,270.630,90.210
0,Load2,217.124,70.246
0,Load3,174.840,58.280
```

每台 DER 有单独 profile，且每一行必须包含 `controller_type`：

| 文件 | 设备/节点 | 必需列 | 控制器 |
| --- | --- | --- | --- |
| `pv1.csv` | `Generator.PV1` / bus1 | `hour,pv_kw,controller_type` | `volt_var` |
| `wind1.csv` | `Generator.Wind1` / bus2 | `hour,wind_kw,controller_type` | `constant_pq` |
| `bess1.csv` | `Storage.BESS1` / bus2 | `hour,bess_kw,controller_type` | `soc_schedule` |
| `ev1.csv` | `Load.EV1` / bus3 | `hour,ev_kw,controller_type` | `constant_pq` |

可选无功列为对应的 `*_kvar`。PV 的 `volt_var` 还使用 `q_limit_kvar`、`v_ref_pu`、`droop_kvar_per_pu`。功率单位为 kW/kvar。

## DER 模块

### 统一接口与职责

所有 DER 均继承 `src/der_model.py` 中的 `DERModel`。QSTS 不需要识别某个具体 DER 的内部逻辑，只对每台已注册设备执行同一组操作：读取该时刻注入、写入 OpenDSS、执行控制更新、推进状态。

```python
class DERModel:
    def _validate_profile(self) -> None:
        """检查时序字段和 controller_type。"""

    def get_injection(self, time_step: int) -> dict[str, object]:
        """返回该时刻的计划 P/Q、控制器类型和控制参数。"""

    def apply_to_opendss(self, dss_interface, injection: dict[str, object]) -> None:
        """将当前 P/Q 命令写入相应 OpenDSS 元件。"""

    def control_step(self, dss_interface, injection: dict[str, object]) -> dict[str, object]:
        """根据已求得的网络状态更新控制器命令。"""

    def advance_state(self, injection: dict[str, object], dt_hours: float) -> None:
        """将本时刻状态推进为下一时刻状态。"""
```

其中 `get_injection()` 和 `control_step()` 处理同一时刻的代数关系；`advance_state()` 专门处理时刻之间的状态传递。

### 当前 DER 设备

| 模型 | OpenDSS 对象 | 节点 | 计划数据 | 控制方式 | 是否跨时段存状态 |
| --- | --- | --- | --- | --- | --- |
| `PVModel` | `Generator.PV1` | bus1 | `pv_kw,pv_kvar` | Volt-VAR | 否 |
| `WindModel` | `Generator.Wind1` | bus2 | `wind_kw,wind_kvar` | 恒 P/Q | 否 |
| `BESSModel` | `Storage.BESS1` | bus2 | `bess_kw,bess_kvar` | SOC 约束的计划功率 | 是，SOC |
| `EVModel` | `Load.EV1` | bus3 | `ev_kw,ev_kvar` | 恒 P/Q 充电负荷 | 否 |

PV、Wind 的正 P 为发电；BESS 的正 P 为放电、负 P 为充电；EV 的正 P 为消费功率。`PVModel`、`WindModel` 写入 `Generator`，`EVModel` 写入 `Load`，`BESSModel` 根据 P 的正负切换 `Storage` 的 `Discharging`、`Charging` 或 `Idling` 状态。

### DER profile 规范

每台设备一份 CSV，每行对应一个仿真时刻；`hour` 和 `controller_type` 是所有 DER 的必需列。

```csv
hour,pv_kw,pv_kvar,controller_type,q_limit_kvar,v_ref_pu,droop_kvar_per_pu
12,400,0,volt_var,100,1.0,1000
```

`controller_type` 当前可取：

| 类型 | 数据要求 | 指令含义 |
| --- | --- | --- |
| `constant_pq` | 计划 P/Q | 直接使用 profile 的 P/Q |
| `volt_var` | 计划 P、`q_limit_kvar`、`v_ref_pu`、`droop_kvar_per_pu` | P 按 profile；Q 由接入母线电压决定 |
| `soc_schedule` | 计划 BESS P/Q | P 先按 SOC 可用能量裁剪，再写入 Storage |

如果 profile 没有对应小时，当前模型回退到零 P/Q。正式研究中应保证每台设备的时刻完整且唯一；当前代码仅显式拒绝不支持的控制器类型，并由 `run.py` 检查四份必需 profile 文件是否存在。

### PV Volt-VAR 控制

PV 的 `control_step()` 在每次 OpenDSS 潮流后读取 bus1 三相电压标幺值的平均值 `Vbus`，并计算：

\[
Q_{PV}=\operatorname{clip}\left[k(V_{ref}-V_{bus}),-Q_{max},Q_{max}\right]
\]

低电压时 (Q_{PV}>0)，PV 注入无功；高电压时 (Q_{PV}<0)，PV 吸收无功。新 Q 与上一轮命令存在差异时，QSTS 将其重新写入 OpenDSS 并再次求解。当前 profile 中 `Qmax=100` kvar、`Vref=1.0` p.u.、`k=1000` kvar/p.u.

### BESS 能量状态与功率边界

BESS 的初始 SOC 为 50%，容量 200 kWh，SOC 下限 20%，充/放电效率均为 95%。其计划功率来自 profile，但实际注入先受能量边界限制：当继续放电将低于 20% SOC，或继续充电将超过 100% SOC 时，`BESSModel` 自动削减该时刻 P。

在控制迭代收敛后，以实际 P 更新状态：

\[
SOC_{t+1}=SOC_t-
\frac{P_{t}\Delta t}{\eta_{dis}E}\times100,\quad P_t>0
\]

\[
SOC_{t+1}=SOC_t+
\frac{|P_{t}|\eta_{ch}\Delta t}{E}\times100,\quad P_t<0
\]

其中当前 \(\Delta t=1\) 小时。下一时刻调用 `get_injection()` 时会读取这个更新后的 SOC，而不是从 profile 重置它。

### 添加新的 DER 或控制器

新增设备应同时完成以下工作：

1. 在 `generate_data.py` 中定义相应 OpenDSS 元件及其接入节点。
2. 生成该设备独立的 profile CSV，包含每时刻的 `controller_type`。
3. 继承 `DERModel`，实现 profile 校验、注入写入与需要的控制/状态逻辑。
4. 在 `run.py` 中实例化模型，并传入与 feeder 一致的 `bus_name`。
5. 如为新控制器，在 `DERModel.SUPPORTED_CONTROLLERS` 中登记类型，并在模型的 `control_step()` 中实现控制方程。

这样，QSTS 主循环不需要为新设备增加类型判断。

## QSTS 与控制器耦合

每个时刻依次：

```text
逐负荷 P/Q + DER 计划 P/Q
        ↓
OpenDSS 潮流求解
        ↓
更新电压相关控制器（当前为 PV Volt-VAR）
        ↓
若 P/Q 改变则再次调用 OpenDSS，直至控制收敛
        ↓
推进 DER 状态并记录结果
```

这是一种外层固定点迭代：网络方程由 OpenDSS 求解，控制方程由 Python 根据已求得母线电压更新，再调用 OpenDSS。PV 控制式为：

```text
Q = clip[droop_kvar_per_pu × (v_ref_pu − Vbus), −q_limit_kvar, q_limit_kvar]
```

它不是把控制方程直接塞入 OpenDSS 的内部方程，但会迭代到 DER P/Q 不再改变；每时刻最多 20 次控制更新。

跨时段状态由 `advance_state()` 传递。当前 PV、Wind、EV 是无状态 profile 回放；BESS 在每个收敛时刻后根据实际充/放电功率、200 kWh 容量、95% 效率与 20% SOC 下限更新 SOC。下一时刻的 BESS 计划功率会受该 SOC 约束。

## 输出

`output/` 包含：

- `qsts_bus_voltages.csv`：每时刻、每个节点的电压标幺值与相角；
- `qsts_system.csv`：总基础负荷 P/Q、控制迭代次数、收敛状态、电压范围、损耗、源端功率、每台 DER 的最终 P/Q 与 BESS 期末 SOC。

## 代码

```text
generate_data.py          # 生成网络、负荷与 DER 时序
run.py                    # 注册 DER 并运行 QSTS
src/der_model.py          # DER 模型、控制器类型、BESS 状态方程
src/opendss_model.py      # OpenDSS 交互、控制迭代与状态汇总
src/qsts.py               # 逐时 QSTS 调度
```

当前 BESS 控制是计划功率加 SOC 安全约束，尚未实现价格优化、MPC、Volt-Watt 或多 DER 协同控制。
