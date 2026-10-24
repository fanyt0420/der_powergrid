# DER 时序建模总说明：提交版本

更新：2026-10-24。本文说明区域—站点两层 SPDE、神经系数 SPDE、输入输出、文件依赖与执行方法。

GitHub 提交范围：17 个 Python／JSON 源码与示例配置，以及本文档；不包含测试、TXT 依赖清单、原始/处理后天气数据、训练权重或运行结果。下文已有成绩及结果路径记录本地完成的实验，新克隆仓库不会自带这些产物，结果报告链接也需要生成相应文件后才能访问。神经流程可从第 9 节开始准备数据、训练，再按第 10 节导出电网算例；旧站点流程需要先重建区域输入。

## 1. 提交内容与使用边界

本次成果是：用区域气象随机场提供共同背景，在站点层加入局地随机场，得到不同位置的辐照、温度与风速，再按每站设备参数生成 PV / Wind 功率，接入原有 IEEE13 QSTS。

区域层用 ERA5 同一区域的小时数据标定；站点经纬度、设备参数和局地随机参数使用用户同意的示例配置。站点层完成了机制、敏感性和潮流检查，尚未用真实多站观测标定。当前分辨率为小时，不能据此宣称分钟级波动或真实电网风险已验证。

最终保留两组电网算例：

- `data/ieee13_spde_sites_v2_local_2000m/`：区域背景加默认局地场，作为主算例。
- `data/ieee13_spde_sites_v2_regional_only/`：相同站点和设备、关闭局地场，作为对照。

对应 QSTS 结果在 `output/` 的同名目录。`output/spde/stations_v2/` 保留六组空间尺度及强度对照和汇总图。两组电网算例各导出一个场景成员，不能将它们解释为全部 96 个成员都已运行潮流。

原有 `ieee13_synth`、`ieee13_unbalanced_der`、ERA5 / measured 算例及其文件保持原状。原有 ARMA / Copula 生成器与这条 SPDE 生成路径并列存在；最终 SPDE 路径没有再串联旧 ARMA / Copula。

## 2. 输入是什么

- `data/raw/era5_regional/`：区域训练所用的下载缓存及来源记录。
- `data/processed/spde/region/`：`weather_hourly.csv.gz`、`sites.csv`、`provenance.json`，供重新标定使用。资料为 2013 年 5—6 月、36 个 ERA5 网格、61 天小时数据；它是再分析数据，并非站点实测。
- `data/processed/spde/regional_driver/model.json`：区域网格、背景系数、残差尺度、选定模型参数、联合驱动和来源信息。
- 同目录的 `weather_ensemble_24h.npz`：已冻结的区域天气集合，供站点程序直接运行；主要潜变量形状为 `[96成员, 24小时, 4变量, 36网格]`。
- `data/config/spde_station_example.json`：9 个假设站点的位置、容量、母线连接模板、PV 朝向/倾角及风机参数。
- `data/config/spde_local_example.json`：局地网格、扩散与衰减、平滑尺度、扰动强度和设备转换配置。
- 原有 `data/generate_data/ieee13_era5.py` 及参考 IEEE13 文件：提供网络、负荷、DER 连接模板和 DSS 导出接口。

直接重建站点成果时读取已冻结的区域输入，无需重新下载或标定。要换区域或历史窗口，应先重新准备数据、拟合，再把新区域目录传给站点程序。

## 3. 用什么方法，为什么这样做

### 区域天气残差场

四个变量依次为辐照晴空指数的 logit 变量、100 m 风矢量 u、v，以及温度。先用训练期拟合的日历和坐标背景去除日变化、趋势，再按训练残差尺度标准化。

各场使用线性随机平流—扩散—衰减模型：

\[
\partial_t z_k=-\mathbf v_k\cdot\nabla z_k+D_k\Delta z_k-\lambda_k z_k+\sigma_k\mathcal S_k\dot W_k.
\]

空间平滑后的随机驱动让相邻位置共享部分扰动；扩散与状态演化产生空间联系和时间延续。四个变量有各自的动力参数，其噪声通过相关矩阵联系。它是统计气象残差模型，没有求解完整大气物理方程。

数值上采用隐式时间步、迎风平流和零通量扩散；用 Kalman 滤波处理观测和估计状态。区域参数只在训练资料上拟合，验证集选择候选动力模型和跨变量耦合强度，最后评估留出的时间与网格。42 天训练、10 天验证、9 天测试；4 个空间网格不参与同化。

使用 SPDE 是因为它能把位置、距离和时间演化直接放入同一模型，网格与站点观测也可分开。线性高斯形式便于核查协方差和数据泄漏。在目前资料规模下，它也提供了参数可解释的基线。联合驱动采用滤波代理估计，并非精确的联合最大似然；测试期没有显示稳定的耦合收益，因此仍须保留独立变量对照。

### 站点局地差异

在每个站点计算背景，并对区域状态作双线性插值，然后叠加较细网格上的局地随机残差：

\[
y_k(s_i,t)=m_k(s_i,t)+a_kH_i z_k(t)+H_i^{\mathrm{local}}\delta_k(t).
\]

粗网格插值本身能产生站间差异，但对同一配电网内很近的站点差异通常很小。局地层用于检查粗网格以下的空间变化。它在 12 km 计算域内使用 13×13 网格，间距 1 km，仍按小时演化；当前局地平流速度为零。

每小时对局地潜变量去除计算域均值。使用实际离散系统的平稳协方差，将站点平均边际标准差匹配到指定值：辐照 logit 0.35、u/v 各 0.75 m/s、温度 0.25°C。这样比较 1 / 2 / 4 km 平滑设置时，不会同时随意改变平均局地方差。另在 2 km 下比较 0.5 / 1 / 1.5 倍扰动强度。

这些数值是配置假设。平滑参数不能当成实测相关长度；边界、扩散、衰减、插值和去均值都会改变相关结构。去均值可能使远端局地残差负相关。局地场与区域场独立并额外增加方差，去均值不能消除尺度间的方差重复，后续需要同步多站资料联合标定。

### 从天气到出力

辐照用有界反变换和站点晴空辐照恢复 GHI，使用 ERBS 分解辐射，再按每站朝向、倾角、温度和容量调用 PVWatts。风速由 u/v 的模得到，按各站轮毂高度与幂律换算，接切入、立方上升、额定平台和切出功率曲线。

局地噪声加在气象变量上，可以把天气差异和设备差异分开。程序还对同一批天气计算统一设备参数的出力，检查设备设置的影响。示例曲线并非厂家实测曲线；地形、遮挡和站点平均气候偏差尚未拟合。

## 4. 文件依赖与输出

执行链为：

`regional_spde.py prepare` → 区域预处理 → `regional_spde.py fit` → 区域模型与集合 → `station_spde.py` → 站点天气/功率及两组算例 → 原有 `run.py` → QSTS 结果。

保留的新增源码各有分工：

- `src/weather_spde.py`：网格、空间插值、SPDE 演化、滤波与数值评分。
- `src/regional_weather.py`、`src/weather_io.py`：区域来源、下载与气象预处理。
- `src/regional_spde.py`：区域背景、参数标定、联合驱动与采样。
- `src/site_weather.py`：站点插值、局地场、天气反变换与设备出力。
- `src/spde_case.py`：把已经算出的分站功率导出到现有网络/时序接口。
- `data/generate_data/regional_spde.py`：区域数据准备与标定入口。
- `data/generate_data/station_spde.py`：站点成果、敏感性对照和算例生成入口。

`output/spde/stations_v2/` 中，`summary.json` 记录配置、种子、来源散列、相关性和聚合指标；`station_sensitivity.png` 展示对照；`sites.csv` 保存站点信息；六个 NPZ 分别保存仅区域场、三种平滑尺度和两种额外强度结果。`raw` / `local_residual` 的维度为 `[96,24,4,9]`，`power_kw` 及物理天气通常为 `[96,24,9]`。NPZ 时间为 UTC 小时中点。

每组电网算例有 `network.json`、`network.dss`、`ieee13_base/`、`load_profiles.csv`、`der_profiles/`、`power_by_site.csv`、`weather_by_site.csv`、`sites.csv`、`profile_meta.json` 等。CSV 与 QSTS 用 UTC 区间起点标记，天气/设备物理量在区间中点评估。当前提交验证范围为 QSTS，导出的参考 RMS 配置未针对这些新设备完成 RMS 验证。

`output/<算例名>/` 保存系统、母线电压、支路、约束和承载能力指标 CSV。`tests/test_weather_spde.py`、`tests/test_regional_spde.py`、`tests/test_site_weather.py` 是保留的数值与数据隔离检查源码，不是阶段试验产物。

## 5. 已运行什么，结果如何

已生成 96 个场景、2013-06-22 的 24 个小时、6 PV + 3 Wind 的站点集合。PV 总容量 550 kW，Wind 总容量 120 kW，两组电网算例使用相同设备。

在默认局地设置下，最近两站约 707 m、最远两站约 7.90 km。辐照潜变量条件相关分别约 0.99605、0.92547；仅区域场时约 0.99999、0.99912。这里先减去每小时集合均值，再汇总成员偏差，避免共同日变化掩盖随机空间差异。这些是生成模型的机制结果，不能作为实测相关性验证。

两组算例均已完成全部 24 步 QSTS。保存结果的电压范围：仅区域场约 0.94135—1.07473 pu，默认局地场约 0.94146—1.07473 pu；收敛不等于满足全部电压约束。

本次清理后，26 项数值与数据隔离检查通过；完整重新生成了六组站点场景，集合数组逐值相同，两组新算例各运行了 24 步 QSTS，结果与已保存结果在 1e-10 容差内一致。检查所用临时算例和输出随后移除。

本次提交清理没有重新标定区域参数，也没有改写上述已保存数值成果。删除阶段文件前，已将最终站点运行所需的区域模型与集合整理到 `processed/spde/regional_driver/`；迁移前后站点输入逐值一致。区域集合文件按原字节保留，`model.json` 内保存原报告散列，可与已有结果的来源散列核对。已有结果中的源码散列保留其生成当时的记录。

## 6. 如何执行

在仓库内层根目录执行，使用已有虚拟环境。只查看最终成果不需要运行生成器。

直接运行现有两组 QSTS（会更新同名输出）：

```powershell
$py = '.\venv\Scripts\python.exe'
& $py run.py --data-dir data/ieee13_spde_sites_v2_regional_only
& $py run.py --data-dir data/ieee13_spde_sites_v2_local_2000m
```

从已冻结区域输入重新生成一组站点成果，使用新目录与前缀，避免覆盖提交结果：

```powershell
& $py data/generate_data/station_spde.py --region data/processed/spde/regional_driver --output output/spde/stations_run01 --name-prefix ieee13_spde_sites_run01
& $py run.py --data-dir data/ieee13_spde_sites_run01_regional_only
& $py run.py --data-dir data/ieee13_spde_sites_run01_local_2000m
```

需要重新标定时，逐步执行并检查上一步成功；这是可选的完整重建流程，计算量大于站点生成：

```powershell
& $py data/generate_data/regional_spde.py prepare --prepared data/processed/spde/region_run01
& $py data/generate_data/regional_spde.py fit --prepared data/processed/spde/region_run01 --output data/processed/spde/regional_driver_run01 --fit-budget 1000 --members 96
& $py data/generate_data/station_spde.py --region data/processed/spde/regional_driver_run01 --output output/spde/stations_run02 --name-prefix ieee13_spde_sites_run02
```

若训练优化未收敛，可在相同数据、划分与预算下加 `--resume` 继续未完成的训练；不能以生成文件判断拟合充分。重新训练会产生验证报告与检查点用于核查，不属于这次保留的旧阶段成果。

数值和数据隔离检查：

```powershell
& $py -m unittest discover -s tests -v
```

## 7. 本次整理与后续替换

已删除早期假设 demo、单独区域阶段算例、HOPE 支线及其下载/预处理、阶段图表、报告、优化检查点、旧阶段说明和计划。最终程序复用的导出函数已独立为 `src/spde_case.py`，不再依赖被删除的演示脚本。区域入口保留 `prepare` / `fit`，移除了旧阶段算例的单独导出入口。

保留的 `raw/era5_regional`、`processed/spde/region` 与 `processed/spde/regional_driver` 分别支持来源追溯、重新标定和直接运行站点生成，具有不同用途。提交文件夹时应一起包含；它们和数值输出大多被 Git 忽略，仅复制源码或仅提交 Git 记录不足以携带完整结果。

得到真实资料后，先替换站点位置和设备参数，再用同步多站天气标定局地方差、时间尺度与空间结构。还需要更长季节、不同天气过程、概率区间校准和更多场景的电网评估。原有 ARMA / Copula 背景说明仍在 `docs/era5_synth_arima_copula.md`。

## 8. 统一比较神经模型与基线

统一入口为 `data/generate_data/benchmark_spde.py`，实验组织、公共评分、诊断及原采样适配函数统一放在 `src/unified_benchmark.py`，相应测试集中在 `tests/test_unified_benchmark.py`。结果集中在 [统一比较报告](../output/spde/benchmark/report.md)。旧 2013 短记录比较和独立的神经双模型图表不再作为并列结果入口。

共同数据为 2015—2019 年、36 个区域 ERA5 网格；2015—2017 年训练、2018 年验证、2019 年评价，中央 4 格留出。每个评价年每月 15 日起报、连续 72 小时、48 个成员，未来不再同化实况。2019 年已经被前期查看，本轮属于统一开发比较，不是新的独立盲测。验证年的 72 小时成绩用于展示，不据此重新选择原神经权重；原权重仍来自原先的 24 小时验证。

协议分开评价，不能混合排名：

- `conditional`：允许使用起报前恰好 24 小时的非留出天气。比较神经 SPDE、原多年固定 SPDE、同固定系数加 Kalman 状态初始化、经验空间 AR1、带训练误差轨迹的持续性、逐小时重采样和连续 72 小时历史块。两种重采样忽略当前历史，作为气候参考。
- `calendar_only`：不读取验证/测试天气作为输入，只给日历和晴空等已知量。神经 SPDE、固定 SPDE、空间 AR1 使用共同的训练期同月份、同小时历史先验；另有逐小时和历史块重采样、原 ARMA/Copula。这里是 72 小时窗口生成，不是全年连续或严格平稳分布的验证。

固定 SPDE 与神经模型沿用原多年数据拟合的权重，没有在这次比较中重新训练。Kalman 对照只改变同一固定系数的状态初始化，不能据此宣称比较了神经与 Kalman 参数训练。空间 AR1 按训练小时转移做 OLS，再拟合空间/跨变量新息协方差，采用固定 5% 对角收缩和 PSD 修正；不是旧空间 OU 的换名。

原 ARMA/Copula 在同一训练期的一个可见参考网格上重新拟合 12 个月，直接调用原完整 AIC 搜索、日 VAR、创新修正及采样器；10 m 风速来自同批季度原始缓存，不用高度近似代替。它没有条件预测接口，所以只参加 `calendar_only`；单站输出共享于区域，不能视为学到了多站空间关系。原 t 新息后仍用正态 CDF、逐日滤波与样本缩放等做法保留并披露。旧 2013 拟合模型及其共同/独立网格旧参数消融不混入这轮排名，因为训练数据和预处理不同。

统一评分包括实际 GHI、100 m 风速和温度的 CRPS、RMSE、90% 覆盖率/宽度与区间分数；全部/留出网格分列，另分 1—6、7—24、25—72 小时时效。整段轨迹 Energy、时间/空间变差、ACF、空间与跨变量相关、低风低光事件和配对窗口 bootstrap 一起报告。尺度、协方差、事件阈值与历史库只用训练年非留出数据。有限成员集合不是独立实况年份。

使用已有模型和匹配基线复算：

```powershell
$project = 'C:\Users\sweetarcher\Desktop\der_powergrid-main\der_powergrid-main'
$py = "$project\venv\Scripts\python.exe"
& $py "$project\data\generate_data\benchmark_spde.py" --resume
```

新数据或配置使用新的 `--output`，用 `--prepared` 指定多年数据、`--models` 指定包含两个权重及预处理/配置的模型目录。`--members`、`--seed` 可改，但不能拿不同协议或设置的成绩直接比较；`--resume` 拒绝来源、代码或参数不匹配。比较不新增电网算例，不验证真实局地站点或电气风险。 夜间辐照潜状态没有观测值：神经/固定 SPDE 使用原先的平滑先验，空间 AR1 使用自身协方差先验；共同的是训练历史和可用信息，不代表各方法隐藏状态先验完全相同。

结果目录的 `marginal_*.csv`、`joint_*.csv`、`structure_*.csv` 分别保存边际、轨迹、相关结构；`events_by_origin.csv` 保存事件，`paired_intervals.csv` 保存配对区间，`training_initializations.csv` 追溯无测试天气的历史来源。`protocol.json` 保存协议与校验值，`linear_baselines.npz`、`baseline_fit.json`、`regional_reference_model.json` 保留复算需要的基线。

本次已删除旧 `output/spde/benchmark_v1/` 的 18 个文件，以及神经目录中 8 个重复的评分、比较图和报告，共 26 个旧比较文件。`output/spde/neural_v1/` 只保留两个权重、两份训练记录、预处理参数、配置/训练来源和示例天气，共 7 个运行/复用文件；训练入口不再生成这些旧比较。原始项目、训练数据与最终电网算例不删。

统一比较还需要保留 `data/raw/era5_neural/` 的训练期季度缓存：原 ARMA/Copula 的真实 10 m 风速从这里读取，缓存模型复用时也校验来源。不能因神经训练只读年度数组，就删除统一比较所需的原始缓存。

本次交付检查：52 项自动检查通过；9 个统一指标 CSV 复算后字节一致；7098 行边际评分的目标支持在各方法间一致，624 行联合评分和 312 行事件评分完整。147 个受保护的原项目、数据、神经权重/训练记录/示例天气及电网算例文件校验值未变。清理后再次运行训练入口，神经目录仍保持上述 7 个文件，不会重建旧比较。

### 8.1 本次统一结果

2019 年条件预测的三变量平均标准化 CRPS：神经 SPDE 0.3698、固定 SPDE 0.4003、空间 AR1 0.3319。神经较固定模型改善约 7.6%，但边际误差没有超过空间 AR1；其空间变差分数约 0.0409，空间 AR1 约 0.0708，二者存在边际与空间结构的取舍。

不读取评价天气的 72 小时窗口生成中，神经平均 CRPS 为 0.4091，原 ARMA/Copula 为 0.3663；空间变差分数分别为 0.0421 与 0.2209。原单站方法边际表现较好，但空间结构误差明显更大。不能据此宣布任何方法在所有目标上最好，也不能将条件预测的优势等同于无条件生成的优势。12 个配对起报窗口的区间及分变量、留出网格成绩以统一报告为准。

## 9. 神经网络学习 SPDE 参数：多年资料分支

已新增同一区域 2015—2019 年 ERA5 数据：36 个网格、43,824 个连续小时。原始下载在 `data/raw/era5_neural/`，按季度缓存；紧凑年度数组和来源记录在 `data/processed/spde/multiyear_2015_2019/`。显式使用 ERA5，并关闭地形降尺度，没有混入不同年份的自动天气产品。

训练年份为 2015—2017，验证年为 2018，测试年为 2019。背景、尺度和网络训练只用训练年非留出网格；4 个中央网格也不进入历史输入与验证选模。神经系数网络有 2,320,422 个参数，每小时根据前 24 小时残差场和日历输出各变量的区域共享平流、扩散、衰减、噪声幅度/长度、平衡偏移和联合驱动因子。完整天气状态仍由隐式 SPDE 推进，没有用网络直接输出天气。

训练使用可见 ERA5 残差作为状态代理的条件似然，配合连续 6/24 小时 CRPS 和时间变差损失；这不同于旧 Kalman 模型的精确线性高斯状态积分。为公平比较，新增了同一算子、数据和训练目标的固定系数 SPDE 对照。当前没有学习逐网格异质系数，也不能称为大规模气象物理求解器。

比较结果统一见第 8 节报告。神经模型的原权重和验证记录保留；不再将旧双模型分数与其他年份的 benchmark 并排。当前不是全年无条件模拟，也未完成真实多站局地参数标定。

新增文件各有分工：

- `src/neural_weather_spde.py`：训练预处理、系数网络、可微 SPDE、联合概率与轨迹损失。
- `data/generate_data/neural_spde_data.py`：多年 ERA5 下载、季度缓存和年度紧凑数据。
- `data/generate_data/train_neural_spde.py`：训练、按验证集选模型并保存示例天气；所有模型比较交给统一 benchmark。
- `requirements-neural.txt`：本地可选训练依赖清单；按本次提交范围不上传，仓库安装方法见下文。
- `tests/test_neural_weather_spde.py`：未来/留出网格隔离、联合噪声协方差、连续潜状态和梯度检查。未安装可选 Torch 时仅跳过此组。

已有虚拟环境中已安装 PyTorch 2.7.1 + CUDA 12.6，并实际使用 RTX 4060 Laptop GPU。新克隆仓库可先安装原有 `requirements.txt`，再补装 `pvlib`、`statsmodels`、`scipy` 和 `torch==2.7.1`；例如在所用 Python 环境运行 `python -m pip install -r requirements.txt`，随后运行 `python -m pip install "pvlib>=0.10,<1" "statsmodels>=0.14,<1" "scipy>=1.10,<2" "torch==2.7.1"`。需要 GPU 时按 PyTorch 官方方式选取与设备匹配的 CUDA wheel；本次不提交或修改 TXT 依赖清单。

从头下载、训练（默认完成五年数据和本轮结构；已有结果目录需换名）：

```powershell
$project = 'C:\Users\sweetarcher\Desktop\der_powergrid-main\der_powergrid-main'
$py = "$project\venv\Scripts\python.exe"
& $py "$project\data\generate_data\neural_spde_data.py"
& $py "$project\data\generate_data\train_neural_spde.py" --output "$project\output\spde\neural_run02"
```

复用已完成模型并重导示例天气，使用 `train_neural_spde.py --resume`；它不再生成独立比较图表或评分。统一评价使用 `benchmark_spde.py --resume`。换年份时，为下载脚本指定 `--start-year`、`--end-year` 和新的 `--output`；最后一年为测试、倒数第二年为验证，其余年份为训练，至少三年。然后把新数据目录传给训练脚本的 `--data` 并使用新的结果目录。已有目录不允许静默改变划分。

`output/spde/neural_v1/` 中的 `neural_best.pt` 是神经模型，`constant_best.pt` 是同条件固定系数对照，训练记录和 `protocol.json` 保存配置及来源。`neural_weather_72h.npz` 是 2019 年 7 月的 48 条区域轨迹，包含潜变量、反变换前变量和物理天气。

`station_spde.py` 仍使用旧区域集合。神经天气现通过独立入口 `neural_spde_case.py` 转换为分站功率和新电网算例，见第 10 节；没有重新训练神经模型，也没有替换原最终站点算例。统一 benchmark 仍属于天气模型验证，不代表已经完成电网风险验证。

本神经分支交付检查：42 项自动检查通过；20 个季度下载的 36 个网格位置一致；两个模型权重和三个主要指标 CSV 在复算后字节一致；113 个受保护的原项目与最终成果文件散列保持不变。

## 10. 将上一版神经天气接入 run.py

已新增转换入口 `data/generate_data/neural_spde_case.py`、转换模块 `src/neural_case.py` 和检查文件 `tests/test_neural_case.py`。这次增加的是天气到电网的接口，没有增加新一版神经网络或校准层；`run.py`、原训练代码、公共站点转换和旧最终算例保持原样。

计算链为：保存的神经区域天气 → 选一个随机成员与连续时段 → 插值到示例站点 → PV/风机功率计算 → 标准算例文件 → 原 `run.py`。

### 10.1 输入、方法与依赖

默认读取：

- `output/spde/neural_v1/neural_weather_72h.npz`：2019 年 7 月 15—17 日、48 个成员、36 网格的连续 72 小时区域天气。
- `data/processed/spde/multiyear_2015_2019/manifest.json` 与 `sites.csv`：核对天气来源、网格位置和节点顺序。转换过程不读取五个年度数组，也不加载网络权重。
- `data/config/spde_station_example.json`：9 个示例站点及设备、电气连接。
- `data/config/spde_local_example.json`：仅借用 PV、风机功率转换设置；不会启用其中的局地随机场参数。

转换使用天气文件中的 `raw`（已撤销标准化、加回背景的 solar_logit、风 u/v、温度），不能把 `latent` 当作原始天气。先按站点经纬度映射到区域网格坐标，重排源节点并作双线性插值，再计算站点晴空辐照、GHI、风速和温度。站点布局原点与区域网格原点分别处理，超出区域的站点直接拒绝，不作外推。

PV 使用原有 pvlib/PVWatts 与辐照分解，风电使用原有轮毂高度换算和功率曲线。完整保留各站容量、倾角、朝向和风机差异；没有重新拟合站点气候或添加局地随机噪声。晴空计算沿用 100 m 海拔假设。负荷沿用原示例的日曲线，跨日重复，并非天气条件下生成的负荷。

导出复用 `src/spde_case.py` 与原 `ieee13_era5.py` 的电气模板，参考 DSS 从项目已有 `data/*/ieee13_base/` 复制。需要保留这些公共源码与至少一套参考 DSS。转换本身无需 PyTorch、旧 Kalman 模型或重新下载天气；运行潮流仍需原 dss-python 环境。

### 10.2 已完成的算例与入口

已导出并实际运行第 0 个成员：`data/ieee13_neural_member000/`，包含完整 72 小时、6 个 PV 和 3 个风电设备。天气文件中点为 00:30—23:30 UTC，导出到功率和潮流文件的标签为对应小时的 00:00—23:00 UTC。

重新运行这个已有算例：

```powershell
$project = 'C:\Users\sweetarcher\Desktop\der_powergrid-main\der_powergrid-main'
$py = "$project\venv\Scripts\python.exe"
& $py "$project\run.py" --data-dir "$project\data\ieee13_neural_member000"
```

选择另一个成员，新建算例并直接运行：

```powershell
& $py "$project\data\generate_data\neural_spde_case.py" --member 1 --run
```

不加 `--run` 时只导出输入文件。成员从 0 开始，当前文件可选 0—47。默认导出全部剩余小时；`--hours 24` 导出 24 小时，`--start-index 24 --hours 24` 选择第二天。若同一成员目录已存在，使用 `--case` 另取一个空目录，例如：

```powershell
& $py "$project\data\generate_data\neural_spde_case.py" --member 1 --start-index 24 --hours 24 --case "$project\data\ieee13_neural_member001_day2" --run
```

入口拒绝覆盖已有算例，不支持把当前 72 小时文件凭空延长。其他保存窗口可通过 `--weather` 指定，并用 `--prepared` 提供相匹配的来源清单和坐标。`--layout` 可替换布局；`--physics` 可替换设备转换配置。`--output` 仅与 `--run` 配合，指定潮流结果目录。

### 10.3 输出与检查

算例目录包含 `network.json`、`network.dss`、参考 DSS、`load_profiles.csv`、9 个 `der_profiles/*.csv`、`sites.csv`、`power_by_site.csv`、`weather_by_site.csv` 和 `profile_meta.json`。元数据记录成员、时段、输入校验值、插值权重和假设。

潮流结果在 `output/ieee13_neural_member000/`：电压、系统功率与网损、支路结果、约束记录和裕度指标，共五个 CSV。一次运行对应一条连续天气轨迹，成员之间不拼接；多成员风险统计需分别运行后汇总。

新增检查验证成员和源节点顺序、不同坐标原点的双线性插值、范围/来源/时间校验，以及导出功率和 `ProfileStore` 的实际读取契约。完整 72 小时潮流已运行；收敛与电压、热约束是否满足应分别查看，不能把运行成功视为没有越限。

本次接口交付检查：46 项自动检查通过；第 0 个成员的 72 个小时全部潮流收敛；9 个设备功率与时间轴完整。112 个受保护的原项目、神经训练结果和原最终算例文件校验值保持不变。接口没有重新训练或替换神经权重。

### Benchmark 文件合并后的复现

公共评分、区域诊断和 ARMA/Copula 采样适配已合并到 `src/unified_benchmark.py`，计算公式保持不变。入口仍为 `data/generate_data/benchmark_spde.py`。合并前结果保留原始源码记录；由于源码校验已变化，重新比较时应使用新的 `--output`（例如 `output/spde/benchmark_merged`），不要在旧目录使用 `--resume`。新目录首次运行完成后可用同一目录配合 `--resume` 重算。数据、训练权重和已有电网算例无需重新生成。
