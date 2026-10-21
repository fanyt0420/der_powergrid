# 三相 GFM/GFL 类 Kuramoto 动态模块

模型以相角与功率耦合为主线，保留电压幅值、功率滤波、GFL 锁相及简化电流响应，通过三相网络联立成 DAE。它比普通 Kuramoto 保留更多设备与网络动态；相较包含详细控制环节的标准 RMS 模型，省略了完整内环和更快的电磁过程。标准 RMS 同样可以描述 GFM/GFL 混合 DAE。本文“RMS 原型”专指仓库中的 `src/rms_model.py` 实现。

代码位于 `src/dynamics/`，入口为 `run_dynamic.py`，测试位于 `test/test_dynamics.py`，动态配置随算例放置。默认配置包含一台储能 GFM、单相光伏 GFL、三相风电 GFL，共 17 个微分状态与 76 个电压代数变量。

设备按实例分配状态，电流映射与相位旋转均使用实际端子顺序。基本测试除了 IEEE 13 节点算例，还覆盖一个双母线、不同相节点排列和不同设备标识的网络，以及三相 GFL 端子重排；这些测试验证接口可按数据装配，不代表已覆盖任意元件、孤岛或大规模性能。

| 复用部分 | 用法 |
|---|---|
| `src/power_flow.py` | 复用 NetworkModel、OperatingPoint 和求解器上下文 |
| `src/profile_store.py` | 重放已有负荷、DER 指令及储能 SOC 到所选断面 |
| `src/solvers/opendss_solver.py` | 生成一致 QSTS 工作点，导出电压、档位与辅助耗电 |
| `src/solvers/fem_elements.py` | 复用线路、变压器、调压器、端子连接及电压基准计算 |

RMS 原型通过 P/Q 指令与网络连接，设备内部角度尚未直接进入注入表达式。本模块需要 GFM 内电势角与 GFL 锁相角直接进入复电流，因此单独实现设备注入和隐式推进，保留 RMS 原型的接口与行为。

`network.py` 是对共享元件的薄装配层。节点顺序来自所选 QSTS 的实际相节点名，调压器使用该断面档位。动态窗口固定源端实际电压，与原生 FEM 潮流可采用的有限源阻抗边界分开处理。当前 DSS 文件的闭合开关使用默认长度，实际阻抗与 JSON 字段有倍率差异；动态适配器仅在局部元件数据中应用 `switch_impedance_multiplier`，不修改共享元件构造器或原始网络。

当前通过 OpenDSS 重放断面并导出参考量，尚不支持从 FEM 保存结果直接启动。

## 代码接口与单位

常规调用使用 `run_case()`；下表中的模型和数值函数供模块内部编排及测试使用，不属于 `PowerFlowSolver` 协议。

| 接口 | 输入与返回 |
|---|---|
| `snapshot.capture_snapshot(network, data_dir, timestamp=None)` | 重放至所选时刻，返回包含节点顺序、实际复电压、档位、负荷与 DER 功率及网络校核参考量的字典；电压为 V，复功率为 VA |
| `model.KuramotoDAE` | 按配置和断面装配设备状态与网络变量；`x` 按设备状态切片，`y` 为非源节点电压标幺值的实部、虚部拼接 |
| `model.evaluate(x, y, **kwargs)` | 返回微分右端、节点复电流注入、设备记录、负荷功率和静态功率；节点电流为 A，设备控制状态使用各设备基值 |
| `model.voltage(y)` / `model.kcl(x, y, **kwargs)` | 前者还原含固定源端的实际电压 V；后者返回按节点电流基值缩放的实数 KCL 残差 |
| `solver.equilibrium(model)` | 返回一致平衡点 `(x, y, info)`；需同时检查求解状态与设备适用域 |
| `solver.step(model, x_old, y_old, dt, **kwargs)` | 联立求一个后向欧拉时间步，返回 `(x, y, info)`，`dt` 单位为 s |
| `solver.algebraic_solve(model, x, y, **kwargs)` | 固定设备状态，重算事件后的网络电压，返回 `(y, info)` |
| `solver.modes(model, x, y)` | 消去电压代数变量后计算小信号模态，返回特征值、频率、阻尼和代数雅可比条件数等记录 |
| `run.run_case(data_dir, output, config, scenario="flat_run", timestamp=None, snapshot=None)` | 执行单场景并写入结果目录，返回运行 manifest；可传入已捕获快照以供多场景共享 |

`run_case()` 的输出目录是场景目录；重复运行会覆盖该目录中的本模块结果文件。配置先校验，运行阶段的数值失败写入 `dae_failure.json`，调用方应检查返回值的 `success`；配置不合法则直接抛出异常。事件通过 `scales` 与 `tripped` 传入模型：前者改变负荷倍数，后者移除设备注入并冻结其微分状态。事件时刻分别保存 `pre_event`、`post_event`，读取 CSV 时应保留这两个同时间记录。

模型使用固定源端电压和窗口内固定档位，动态 DER 采用星形接线。新增网络或设备需核对元件与接线支持范围。`src/dynamics/solver.py` 负责动态联立求解，`src/solvers/` 中的后端实现单个工作点的潮流求解。

## 运行

```bash
python run_dynamic.py --scenario load_step
python run_dynamic.py --scenario all
python run_dynamic.py --timestamp "2026-01-01 12:00:00" --scenario pv_trip
python run_dynamic.py --data-dir data/ieee13_unbalanced_der --config data/ieee13_unbalanced_der/dynamic_config.json --scenario flat_run
python -m unittest test.test_dynamics -v
```

`--output` 指定场景目录，选择 `all` 时在该目录下按场景分子目录。每次保存状态、电压、设备响应、残差、初始化、模态、输入参数和源码散列。默认步长为 0.005 s，计算窗口 2 s，缩放残差门槛为 1e-7；电流越限时停止。数值完成不等于稳定性或步长精度通过验收。

## 对照图

运行 `python -m src.dynamics.demo`，使用 12:00 光伏和风电均有功输出的断面，在 1 s 对 671 负荷增加 10% 的有功、无功。两组计算仅改变储能 GFM 输出电抗，取 0.073 与 0.20 pu，分别重新求一致平衡点和模态，使用相同网络、功率目标及扰动。电抗取值是研究工况，不是设备辨识或推荐整定值。

每组计算采用 0.0025 与 0.00125 s 两个步长，频率面板同时给出两种步长的轨迹，便于观察数值耗散；峰值相对差记录在 `comparison.json`，供判断步长影响。其他面板分别提供全网逐相电压热图、GFM 有功响应和 GFL 锁相误差。

响应曲线及热图列标题用橙色表示工况 A（$x_c=0.073$ pu）、蓝色表示工况 B（$x_c=0.20$ pu），颜色图例集中放在图顶。线型只在频率面板区分步长：实线为 0.00125 s，点线为 0.0025 s。热图使用细步长，覆盖 38 个非源相节点；每行对应一个母线相别，同母线以分隔线分组。两列共用电压色标，值为电压幅值相对扰动前瞬间的变化，单位为 $10^{-3}$ pu，热图颜色不表示工况类别。

绘图会校验运行成功、断面时间、输入散列、对照参数及数据时间网格，避免混用不同工况的结果。已有记录可用 `python -m src.dynamics.demo --plot-only` 重绘，无需重新运行动态仿真。

结果写入 `output/dynamic_demo/dynamic_comparison.png` 和同名 PDF。PDF 是一张 A3 横向综合图，包含左侧双工况热图和右侧三类设备响应；文字、坐标轴和曲线保留矢量形式。原始运行记录保留在相应子目录。该示例展示参数与耦合的关系，不作为真实设备稳定性或承载力提升结论。
