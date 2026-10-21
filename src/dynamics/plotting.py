"""Spatial voltage response and complementary device responses in one PDF."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd


def plot(summary, output, gfm_id, gfl_device, network):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 11,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    reactances = sorted({r["x_c_pu"] for r in summary["runs"]})
    steps = sorted({r["dt_s"] for r in summary["runs"]})
    if len(reactances) != 2 or len(steps) != 2:
        raise ValueError("The comparison requires two reactances and two timesteps")
    colors = dict(zip(reactances, ("#D55E00", "#0072B2")))
    names = dict(zip(reactances, ("A", "B")))
    fine_dt, coarse_dt = steps
    event_time = summary["event"]["time_s"]
    source = str(network.base["slack_bus"]).lower()
    nodes = [(str(bus["id"]).lower(), p) for bus in network.buses
             if str(bus["id"]).lower() != source for p in sorted(bus["phases"])]
    keys = ["time_s", "event_side"]
    data, heatmaps = [], {}
    reference_inputs, reference_config = None, None

    for run in summary["runs"]:
        path = Path(run["output"])
        manifest = json.loads((path / "dae_manifest.json").read_text(encoding="utf-8"))
        if not manifest["success"] or manifest["qsts_timestamp"] != summary["timestamp"]:
            raise ValueError(f"Inconsistent or failed run: {path}")
        cfg = copy.deepcopy(manifest["config"])
        if cfg.pop("dt_s") != run["dt_s"]:
            raise ValueError(f"Timestep differs from comparison: {path}")
        varied = next(d for d in cfg["dynamic_devices"] if d["device_id"] == gfm_id)
        if varied.pop("x_c_pu") != run["x_c_pu"]:
            raise ValueError(f"Reactance differs from comparison: {path}")
        if cfg["scenarios"]["load_step"] != [summary["event"]]:
            raise ValueError(f"The figure requires one recorded load-step event: {path}")
        if reference_config is None:
            reference_config, reference_inputs = cfg, manifest["input_sha256"]
        elif cfg != reference_config or manifest["input_sha256"] != reference_inputs:
            raise ValueError(f"Additional parameters or input data changed: {path}")
        states = pd.read_csv(path / "dae_state.csv")
        t = states.time_s.to_numpy()
        mask = t >= event_time
        entry = {"run": run, "time": t[mask]-event_time,
                 "frequency": states[f"{gfm_id}_delta_f_hz"].to_numpy()[mask]}
        data.append(entry)
        if run["dt_s"] != fine_dt:
            continue
        der = pd.read_csv(path / "dae_der.csv")
        device = der[der.device_id == gfm_id]
        if not device[keys].reset_index(drop=True).equals(states[keys].reset_index(drop=True)):
            raise ValueError(f"GFM and state timestamps differ: {path}")
        entry["power"] = (device.p_kw.to_numpy()-device.p_kw.iloc[0])[mask]
        voltage = pd.read_csv(path / "dae_bus_voltage.csv", dtype={"bus":str})
        phase = gfl_device.get("terminal_nodes", gfl_device["phases"])[0]
        ref = voltage[(voltage.bus == str(gfl_device["bus"]).lower()) & (voltage.node == phase)]
        if not ref[keys].reset_index(drop=True).equals(states[keys].reset_index(drop=True)):
            raise ValueError(f"GFL voltage and state timestamps differ: {path}")
        angle = np.deg2rad(ref.angle_deg.to_numpy()) - {1:0, 2:-2*np.pi/3, 3:2*np.pi/3}[phase]
        theta = states[f"{gfl_device['id']}_theta_pll_rad"].to_numpy()
        entry["pll"] = np.rad2deg(np.angle(np.exp(1j*(angle-theta))))[mask]
        pre = voltage[(voltage.time_s == event_time) & (voltage.event_side == "pre_event")]
        if len(pre) == 0:
            raise ValueError(f"Missing event-left voltage reference: {path}")
        baseline = pre.set_index(["bus","node"]).v_pu.reindex(nodes)
        post = voltage[(voltage.time_s >= event_time) & (voltage.event_side != "pre_event")]
        table = post.pivot(index=["bus","node"], columns="time_s", values="v_pu").reindex(nodes)
        if baseline.isna().any() or table.isna().any().any():
            raise ValueError(f"Missing phase-node samples for voltage heatmap: {path}")
        times = table.columns.to_numpy(dtype=float)-event_time
        values = (table.to_numpy()-baseline.to_numpy()[:,None])*1000
        heatmaps[run["x_c_pu"]] = (times, values)

    # A3 landscape: spatial information occupies the left column, device
    # responses the right. The heatmaps share one physical colour scale.
    fig = plt.figure(figsize=(16.54, 11.69))
    grid = fig.add_gridspec(3,2,left=.07,right=.975,bottom=.235,top=.80,
                           width_ratios=(1.12,1),hspace=.52,wspace=.27)
    hgrid = grid[:,0].subgridspec(1,2,wspace=.10)
    heat_axes = [fig.add_subplot(hgrid[0,i]) for i in range(2)]
    response_axes = [fig.add_subplot(grid[i,1]) for i in range(3)]
    low = np.floor(min(0, *(a.min() for _,a in heatmaps.values())))
    high = np.ceil(max(0, *(a.max() for _,a in heatmaps.values())))
    if high-low < 1e-12:
        high = low+1e-12
    norm = Normalize(vmin=low, vmax=high)
    labels = [f"{bus.upper() if bus=='rg60' else bus}.{'ABC'[phase-1]}" for bus,phase in nodes]
    boundaries = [i-.5 for i in range(1,len(nodes)) if nodes[i][0]!=nodes[i-1][0]]
    for ax,xc in zip(heat_axes,reactances):
        times, values = heatmaps[xc]
        edges = np.r_[times[0],(times[:-1]+times[1:])/2,times[-1]]
        image = ax.pcolormesh(edges,np.arange(len(nodes)+1)-.5,values,
                               cmap="magma",norm=norm,rasterized=True,shading="flat")
        ax.set_ylim(len(nodes)-.5,-.5)
        ax.set_title(f"工况 {names[xc]}",color=colors[xc],fontsize=13,pad=8)
        ax.set_xlabel("扰动后时间 / s")
        ax.set_xticks([0,.5,1])
        ax.set_xlim(times[0],times[-1])
        ax.set_yticks(np.arange(len(nodes)))
        ax.set_yticklabels(labels if ax is heat_axes[0] else [],fontsize=8.5)
        ax.tick_params(axis="y",length=0)
        for boundary in boundaries:
            ax.axhline(boundary,color="white",lw=.55,alpha=.65)
    heat_axes[0].set_ylabel("母线.相别（同一母线以白线分组）",labelpad=10)
    cax = fig.add_axes([.11,.172,.35,.016])
    colorbar = fig.colorbar(image,cax=cax,orientation="horizontal")
    colorbar.set_ticks(np.linspace(low,high,5))
    colorbar.set_label("电压幅值变化 Δ|V| / (10⁻³ pu)，相对扰动前瞬间",fontsize=10.5)
    fig.text(.07,.842,f"(a) 全网逐相电压响应：{len(nodes)} 个非源相节点，共用色标",fontsize=14)

    for entry in data:
        run = entry["run"]
        color = colors[run["x_c_pu"]]
        fine = run["dt_s"] == fine_dt
        response_axes[0].plot(entry["time"],entry["frequency"],color=color,
                              ls="-" if fine else ":",lw=1.6 if fine else 1)
        if fine:
            response_axes[1].plot(entry["time"],entry["power"],color=color,lw=1.6)
            response_axes[2].plot(entry["time"],entry["pll"],color=color,lw=1.6)
    titles = ["(b) GFM 频率偏差：时间演化", "(c) GFM 有功变化：功率交换", "(d) GFL 锁相误差：电压跟踪"]
    ylabels = ["Δf / Hz", "相对扰动前变化 / kW", "端口相位 − PLL 相位 / °"]
    for ax,title,ylabel in zip(response_axes,titles,ylabels):
        ax.set(title=title,ylabel=ylabel,xlabel="扰动后时间 / s")
        ax.grid(alpha=.17)
    handles = [Line2D([],[],color=colors[xc],lw=2.3,
                label=f"工况 {names[xc]}：$x_c$={xc:g} pu") for xc in reactances]
    handles += [Line2D([],[],color="#475569",ls=style,lw=1.6,
                 label=f"仅(b)：Δt={dt:g} s") for style,dt in (("-",fine_dt),(":",coarse_dt))]
    fig.suptitle("类 Kuramoto 模型：三相电压分布与 GFM/GFL 耦合响应",fontsize=21,y=.975)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.936),ncol=4,
               frameon=False,fontsize=12)
    event = summary["event"]
    change = f"P {100*(event['p_multiplier']-1):+g}%、Q {100*(event['q_multiplier']-1):+g}%"
    fig.text(.52,.881,f"QSTS：{summary['timestamp']}  |  t={event_time:g} s，{event['device_id']} 负荷 {change}  |  仅改变储能 GFM 输出电抗",
             ha="center",fontsize=12,color="#475569")
    differences = "；".join(f"工况 {names[r['x_c_pu']]}：{100*r['relative_difference']:.1f}%"
                             for r in summary["peak_step_differences"])
    fig.text(.07,.099,"热图颜色衡量电压变化；列标题和响应曲线颜色区分参数工况。热图及(c)(d)均使用细步长。",fontsize=11)
    fig.text(.07,.065,f"两步长频率峰值相对差：{differences}。增长工况的峰值尚未收敛，时域曲线用于定性对照。",fontsize=11)
    fig.text(.07,.031,"各工况从对应的一致平衡点出发。研究参数对照用于观察耦合动态，不作为实测设备结论。",fontsize=10.5,color="#475569")
    for suffix in ("png","pdf"):
        fig.savefig(Path(output)/f"dynamic_comparison.{suffix}",dpi=220,facecolor="white")
    plt.close(fig)
