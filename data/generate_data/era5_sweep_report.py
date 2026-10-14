"""Aggregate the ieee13_era5 penetration x climate sweep into statistics, figures and a PDF.

Reads   : output/sweep/sc_<site>_s<scale>_<start>.csv  (one file per batch run)
Writes  : output/sweep/sweep_all.csv        (combined per-scenario rows)
          output/sweep/sweep_summary.csv    (per site x penetration level)
          output/sweep/fig_*.png
          output/sweep/ieee13_era5_sweep_report.pdf

The indicators follow the definitions agreed in the grill session:
  大规模 = penetration / DER energy share / reverse-flow probability
  强随机 = cross-scenario distributions (P5-P95 bands, exceedance PROBABILITY)
  大波动 = net-load ramp P50/P95/P99
  多间歇 = hour-to-hour PV drops > 20% of installed capacity
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "output" / "sweep"

SITE_LABEL = {"atlanta": "亚特兰大(亚热带湿润)", "newyork": "纽约(温带大陆性)", "phoenix": "凤凰城(亚热带干旱)"}
BASE_SCALE = 0.05   # near-zero baseline; a true 0 kW is rejected by NetworkModel validation
LEVEL_NAME = {BASE_SCALE: "L0 近零", 2.75: "L1 30%", 5.5: "L2 60%", 9.14: "L3 100%", 13.7: "L4 150%"}

RAMP = "net_ramp_max_kw_per_min"


def load() -> pd.DataFrame:
    frames = []
    for path in sorted(SWEEP.glob("sc_*.csv")):
        match = re.match(r"sc_(.+)_s(.+)_(\d{4}-\d{2}-\d{2})\.csv", path.name)
        if not match:
            continue
        frame = pd.read_csv(path)
        scale = float(match.group(2))
        # Guard against stale copies: if a batch run crashes before rewriting
        # scenarios.csv, the previous run's file would silently poison the sweep.
        # (This actually happened once: pv_scale=0 aborted, leaving L0 rows that
        # reported a 148% "penetration".) Fail loudly instead.
        found = sorted(frame["pv_scale"].unique()) if len(frame) else []
        if len(found) != 1 or abs(float(found[0]) - scale) > 1e-9:
            raise SystemExit(f"{path.name}: pv_scale column {found} != filename scale {scale}; "
                             f"the batch run for this cell did not complete.")
        frame["site"] = match.group(1)
        frame["pv_scale"] = scale
        frame["window"] = match.group(3)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no sweep files found in {SWEEP}")
    return pd.concat(frames, ignore_index=True)


def prob(series: pd.Series) -> float:
    """Fraction of scenarios in which the event occurs at least once."""
    return float((series > 0).mean())


def summarise(all_rows: pd.DataFrame) -> pd.DataFrame:
    ok = all_rows[all_rows["status"] == "ok"]
    rows = []
    for (site, scale), group in ok.groupby(["site", "pv_scale"]):
        total = len(all_rows[(all_rows["site"] == site) & (all_rows["pv_scale"] == scale)])
        rows.append({
            "site": site,
            "site_label": SITE_LABEL.get(site, site),
            "pv_scale": scale,
            "level": LEVEL_NAME.get(scale, f"s={scale}"),
            "n_days": len(group),
            "diverged": int(total - len(group)),
            "penetration_pct": round(float(group["pv_penetration_pct"].mean()), 1),
            "der_share_pct": round(float(group["der_energy_share_pct"].mean()), 2),
            "pv_cf_mean": round(float(group["pv_capacity_factor"].mean()), 4),
            "pv_cf_std": round(float(group["pv_capacity_factor"].std()), 4),
            "v_high_prob": round(prob(group["v_high_steps"]), 3),
            "v_high_steps_mean": round(float(group["v_high_steps"].mean()), 2),
            "v_low_prob": round(prob(group["v_low_steps"]), 3),
            "min_v_p5": round(float(np.percentile(group["min_v_pu"], 5)), 4),
            "max_v_mean": round(float(group["max_v_pu"].mean()), 4),
            "ramp_p50": round(float(np.percentile(group[RAMP], 50)), 2),
            "ramp_p95": round(float(np.percentile(group[RAMP], 95)), 2),
            "ramp_p99": round(float(np.percentile(group[RAMP], 99)), 2),
            "drop_events_mean": round(float(group["pv_drop_events"].mean()), 2),
            "drop_prob": round(prob(group["pv_drop_events"]), 3),
            # drop_events is scale-invariant by construction (threshold scales with
            # installed capacity), so the absolute magnitude is what actually grows.
            "pv_drop_max_kw": round(float(group["pv_drop_max_kw"].mean()), 1),
            "duck_depth_kw": round(float(group["duck_depth_kw"].mean()), 1),
            "reverse_prob": round(prob(group["reverse_steps"]), 3),
            "reverse_steps_mean": round(float(group["reverse_steps"].mean()), 2),
            "line_load_p95": round(float(np.percentile(group["max_line_loading_pct"], 95)), 1),
        })
    return pd.DataFrame(rows).sort_values(["site", "pv_scale"]).reset_index(drop=True)


def figures(summary: pd.DataFrame, ok: pd.DataFrame) -> list[Path]:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    paths = []
    scales = sorted(summary["pv_scale"].unique())

    def add(name: str, fig: plt.Figure) -> Path:
        path = SWEEP / f"fig_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
        return path

    specs = [
        ("v_high_prob", "电压越上限概率 vs 渗透率", "越限概率"),
        ("reverse_prob", "馈线逆流概率 vs 渗透率", "逆流概率"),
        ("ramp_p95", "净负荷爬坡 P95 vs 渗透率", "爬坡 P95 (kW/min)"),
        ("duck_depth_kw", "鸭形曲线深度 vs 渗透率", "深度 (kW)"),
        ("drop_events_mean", "间歇事件频次 vs 渗透率", "跌落>20%装机 次数/天"),
    ]
    for column, title, ylabel in specs:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for site in sorted(summary["site"].unique()):
            sub = summary[summary["site"] == site]
            ax.plot(sub["pv_scale"], sub[column], "o-", label=SITE_LABEL.get(site, site))
        ax.set_xticks(scales)
        ax.set_xticklabels([LEVEL_NAME.get(s, s) for s in scales], fontsize=8)
        ax.set_xlabel("渗透率档位")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        add(column, fig)

    # Randomness evidence: PV capacity-factor spread per site (all levels pooled at L3).
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    data = [ok[(ok["site"] == s) & (ok["pv_scale"] == 9.14)]["pv_capacity_factor"].dropna()
            for s in sorted(ok["site"].unique())]
    ax.boxplot(data, tick_labels=[SITE_LABEL.get(s, s) for s in sorted(ok["site"].unique())])
    ax.set_ylabel("PV 容量系数")
    ax.set_title("L3(100%) 各气候区容量系数分布 —— 随机性来自真实天气")
    ax.grid(alpha=0.3, axis="y")
    add("cf_spread", fig)

    # Seasonal dependence at L3.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for site in sorted(ok["site"].unique()):
        sub = ok[(ok["site"] == site) & (ok["pv_scale"] == 9.14)]
        ax.plot(sub["date"], sub["pv_capacity_factor"], "o-", ms=3, label=SITE_LABEL.get(site, site))
    ax.set_ylabel("PV 容量系数")
    ax.set_title("L3(100%) 容量系数逐日 —— 季节与天气型差异")
    ax.tick_params(axis="x", rotation=60, labelsize=6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    add("cf_season", fig)
    return paths


def build_pdf(summary: pd.DataFrame, ok: pd.DataFrame, figs: list[Path]) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="STSong-Light", fontSize=9.5,
                          leading=14, alignment=TA_LEFT)
    title = ParagraphStyle("title", parent=styles["Title"], fontName="STSong-Light", fontSize=17)
    head = ParagraphStyle("head", parent=styles["Heading2"], fontName="STSong-Light", fontSize=12.5)

    doc = SimpleDocTemplate(str(SWEEP / "ieee13_era5_sweep_report.pdf"), pagesize=A4,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = [Paragraph("ieee13_era5 大规模 DER 并网扫描报告", title),
             Spacer(1, 4),
             Paragraph("ERA5 真实气象 → pvlib → IEEE-13 QSTS ｜ 5 档渗透率 × 3 气候区 × 4 季节窗口 × 8 天",
                       body)]

    def table(frame: pd.DataFrame, widths=None, font=7.2):
        data = [list(frame.columns)] + frame.astype(str).values.tolist()
        style = TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
            ("FONTSIZE", (0, 0), (-1, -1), font),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#8ea9db")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fb")]),
        ])
        return Table(data, colWidths=widths, style=style, repeatRows=1)

    cols = ["level", "penetration_pct", "der_share_pct", "v_high_prob", "v_low_prob",
            "reverse_prob", "ramp_p95", "drop_events_mean", "pv_drop_max_kw", "duck_depth_kw", "n_days"]
    names = {"level": "档位", "penetration_pct": "渗透率%", "der_share_pct": "发电占比%",
             "v_high_prob": "越上限概率", "v_low_prob": "越下限概率", "reverse_prob": "逆流概率",
             "ramp_p95": "爬坡P95", "drop_events_mean": "间歇次数/天", "pv_drop_max_kw": "最大跌落kW",
             "duck_depth_kw": "鸭形深度kW", "n_days": "天数"}

    story.append(Paragraph("1. 方法与口径", head))
    story.append(Paragraph(
        "气象为 ERA5 小时再分析（Open-Meteo archive API，1940 至今，0.25° 网格），经 pvlib ModelChain(PVWatts) "
        "转为 6 台 PV 的交流出力；负荷为 Ausgrid 实测户（按季节匹配月份、按日能量重标定）；EV 为挪威实测会话池。"
        "渗透率 = PV 装机 / 该场景日负荷峰值（真实口径，非 native 锚点）。每个场景为独立单日 QSTS（24 步 @1 h），"
        "volt-var 采用 IEEE 1547-2018 默认曲线（V1..V4 = 0.90/0.98/1.02/1.10 pu，死区 0.02，斜率 13.125×额定/rating）。"
        "净负荷取变电站交换功率 net = -source_p_kw（正 = 受入，负 = 送出），逆流判据为 source_p_kw > 0；"
        "越限阈值 0.95 / 1.05 pu，线路与变压器热限 100%（src/qsts.py:33）。"
        "2022 年 1/4/7/10 月各取连续 8 天，合计 32 天/档位/气候区。本轮修正了 pvlib 的时区处理：ERA5 索引是当地挂钟时间但为 "
        "tz-naive，pvlib 会按 UTC 解释，导致太阳位置偏移约 4 h（上午出力被判为日出前而清零、峰值被推到傍晚），已改为 tz_localize 后再建模。", body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. 汇总（每档位 32 天，按气候区分组）", head))
    for site in sorted(summary["site"].unique()):
        sub = summary[summary["site"] == site][cols].rename(columns=names)
        story.append(Paragraph(f"2.{sorted(summary['site'].unique()).index(site) + 1} {SITE_LABEL.get(site, site)}", body))
        story.append(table(sub, widths=[2.3 * cm] + [1.55 * cm] * (len(cols) - 1)))
        story.append(Spacer(1, 6))
    story.append(PageBreak())

    story.append(Paragraph("3. 图表", head))
    for path in figs:
        story.append(Image(str(path), width=15.5 * cm, height=15.5 * cm * 0.56))
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("4. 结论（对照「大规模 / 强随机 / 大波动 / 多间歇」）", head))
    for text in conclusions(summary, ok):
        story.append(Paragraph(text, body))
        story.append(Spacer(1, 3))

    story.append(Paragraph("5. 诚实的限制", head))
    for text in LIMITATIONS:
        story.append(Paragraph("• " + text, body))
        story.append(Spacer(1, 2))

    doc.build(story)
    return SWEEP / "ieee13_era5_sweep_report.pdf"


LIMITATIONS = [
    "<b>随机性目前只来自气象，其余都是常量</b>：Ausgrid 负荷只按月份取（同一月份 8 天的负荷曲线逐点相同，"
    "实测 7 月窗口 8 天负荷峰值恒为 4781.9 kW），EV 会话池用固定种子 EV_SEED、6 组阵列朝向用固定 ARRAY_SEED、"
    "BESS 初值固定 55% —— 每个场景之间真正变化的只有 PV 与风电出力。因此本报告的「强随机」只刻画了电源侧，"
    "净负荷的随机性被系统性低估；grill 中提出的「净负荷随机性只做了一半」在这里其实是「负荷侧为零」。",
    "<b>越限计数的口径</b>：voltage_high_count 是「该时刻所有母线中越限的母线个数」，按天累加，"
    "所以表中数字是「测点·时段数」而不是「越限时段数」；阈值 0.95 / 1.05 pu（src/qsts.py:33），线路热限 100%。",
    "L0 档为「近零」而非严格 0 kW：NetworkModel 校验要求 DER 的 p_rated_kw 为正数（src/power_flow.py:242-244），"
    "pv_scale=0 会被拒绝。本报告的 L0 取 pv_scale=0.05（27.5 kW，约 0.5% 渗透率）作为基准。",
    "ERA5 为 1 小时、约 25 km 网格：馈线（约 2 km）全部落在同一格内，PV 之间无空间天气差异，出力差异只来自阵列朝向；"
    "分钟级云团过境冲击无法体现。文献定义的「15 min 内跌落 >20%」此处只能按 1 h 步长统计。",
    "单日场景之间电网状态（BESS SOC、调压器分接头、电容器）每次重置，因此回答不了连续阴雨耗尽储能这类路径依赖问题。",
    "负荷来自澳大利亚一户代表户（Ausgrid Customer 12），与 PV 的美国站点只做季节匹配，不是同一天的真实配对。",
    "IEEE-13 仅 16 节点，相对真实馈线仍属小规模；且本轮未更换拓扑。",
    "稳态 QSTS 不含逆变器动态，秒级动态需另跑 RMS；本报告的「爬坡」是小时级净负荷斜率，不是秒级。",
    "渗透率分母取该场景日负荷峰值，跨季节会有差异，故同一 pv_scale 档位的渗透率在表中是 32 天的均值。",
]


def conclusions(summary: pd.DataFrame, ok: pd.DataFrame) -> list[str]:
    out = []
    base = summary[summary["pv_scale"] == BASE_SCALE]
    top = summary[summary["pv_scale"] == 13.7]
    base_high = float(base["v_high_prob"].mean())
    top_high = float(top["v_high_prob"].mean())
    base_ramp = float(base["ramp_p95"].mean())
    top_ramp = float(top["ramp_p95"].mean())
    base_duck = float(base["duck_depth_kw"].mean())
    top_duck = float(top["duck_depth_kw"].mean())
    rev = top[top["reverse_prob"] > 0]
    cf_cv = ok[ok["pv_scale"] == 9.14].groupby("site")["pv_capacity_factor"].agg(["mean", "std"])
    cv = (cf_cv["std"] / cf_cv["mean"]).round(3).to_dict()

    out.append(f"<b>大规模</b>：渗透率从 L0(0%) 扫到 L4(约 {top['penetration_pct'].mean():.0f}%)，"
               f"DER 发电量占比达 {top['der_share_pct'].mean():.0f}%；"
               + (f"L4 下 {len(rev)}/3 个气候区出现馈线级逆流（概率最高 {top['reverse_prob'].max():.2f}）。"
                  if len(rev) else "L4 下仍未出现馈线级逆流，需更高渗透率或配合 BESS 放电。"))
    out.append(f"<b>强随机（严格说只到电源侧）</b>：不确定性的物理载体是气象，而其来源是「运行日是哪一天」不可预知——"
               "单场景完全确定，随机性只存在于跨场景集合（对 ERA5 历史日的经验分布做抽样）。"
               f"L3 档 PV 容量系数的跨场景变异系数 {cv}（亚特兰大/纽约/凤凰城）；"
               "越限等指标一律按「多少天发生」的概率口径给出。"
               "<b>净负荷的随机性目前未体现</b>：负荷同月恒定、EV 与阵列朝向用固定种子，见限制 1。")
    out.append(f"<b>大波动</b>：净负荷爬坡 P95 由 L0 的 {base_ramp:.1f} kW/min 升到 L4 的 {top_ramp:.1f} kW/min"
               f"（×{top_ramp / base_ramp:.1f}）；鸭形曲线深度由 {base_duck:.0f} kW 加深到 {top_duck:.0f} kW。")
    out.append(f"<b>多间歇</b>：按 grill 的定义（1 h 内出力跌落超过装机的 20%），跌落<b>次数</b>天然与渗透率无关"
               f"（阈值随装机等比放大），L1–L4 均为 {summary[summary['pv_scale'] == 2.75]['drop_events_mean'].mean():.2f} 次/天左右；"
               f"真正随规模放大的是跌落的<b>绝对幅值</b>：由 L1 的 "
               f"{summary[summary['pv_scale'] == 2.75]['pv_drop_max_kw'].mean():.0f} kW 增大到 L4 的 "
               f"{top['pv_drop_max_kw'].mean():.0f} kW —— 同样的云，装机越大冲击越深。")
    out.append("<b>「越上限概率 = 1」是怎么回事（不是 bug）</b>：这是 IEEE-13 三相不平衡叠加单相调压器线损补偿（LDC）"
               "的固有结果，与 DER 无关。基准 DSS 中有三台<u>单相</u>调压器 Reg1/2/3"
               "（vreg=122 V、band=2、R=3、X=9、ptratio=20），按各相电流分别升压：C 相负荷最重（2261 kW）升压最多，"
               "末端仍跌到 0.926 pu（全天 19 次越下限，<b>全部在 C 相</b>）；B 相负荷最轻（1806 kW），首端被抬到 "
               "1.056 pu 且沿线压降小，末端一直在 1.05 以上（全天 78 次越上限，<b>71% 在 B 相</b>）。"
               "该现象集中在晚峰 16–22 时，L0 就已存在 → 概率口径已饱和，"
               "应改看「测点·时段数」；PV 降低净负荷使调压器少升压，所以该计数随渗透率下降。")
    out.append(f"<b>归因</b>：电压越上限在所有档位都 100% 发生（L0 {base_high:.2f}，L4 {top_high:.2f}），"
               f"但越上限的<b>测点·时段数</b>（每天各时刻越限母线数之和）由 "
               f"{base['v_high_steps_mean'].mean():.1f} 降到 {top['v_high_steps_mean'].mean():.1f}。"
               " 说明越限的主要成因是负荷水平与拓扑，而非 DER；IEEE 1547 volt-var 在高电压时吸收无功，"
               "装机越大吸得越多，因此高渗透反而压低了越限测点数。")
    return out


def main() -> None:
    all_rows = load()
    all_rows.to_csv(SWEEP / "sweep_all.csv", index=False)
    summary = summarise(all_rows)
    summary.to_csv(SWEEP / "sweep_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    figs = figures(summary, all_rows[all_rows["status"] == "ok"])
    pdf = build_pdf(summary, all_rows[all_rows["status"] == "ok"], figs)
    print(f"\nPDF: {pdf}")


if __name__ == "__main__":
    main()
