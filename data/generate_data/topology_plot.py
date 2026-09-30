"""Shared network-topology visualisation for generated QSTS cases."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEVICE_STYLE = {
    "pv": ("PV", "#f6c445"),
    "wind": ("Wind", "#73bfe2"),
    "bess": ("BESS", "#a98bd4"),
    "ev": ("EV", "#e88873"),
}


def plot_network_topology(
    network: Mapping,
    output_file: str | Path,
    positions: Mapping[str, tuple[float, float]],
) -> None:
    """Save a labelled feeder diagram from the case's ``network.json`` data.

    Node fill indicates source/load bus type.  The coloured strips inside each
    node list the connected DERs; each branch is labelled with its ID and length.
    ``positions`` is deliberately supplied by each case generator, so the diagram
    remains readable for radial feeders and for later non-radial examples.
    """
    output_file = Path(output_file)
    devices_by_bus: dict[str, list[Mapping]] = {}
    for device in network["devices"]:
        devices_by_bus.setdefault(str(device["bus"]), []).append(device)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for branch in network["branches"]:
        start, end = str(branch["from_bus"]), str(branch["to_bus"])
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        kind = str(branch.get("kind", "line"))
        color, style = ("#9b59b6", "--") if kind in {"transformer", "regulator"} else ("#c0392b", ":") if kind == "switch" else ("#566573", "-")
        ax.plot([x1, x2], [y1, y2], color=color, linestyle=style, linewidth=2.0, zorder=1)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        detail = f"\n{branch['length_km']:.2f} km" if "length_km" in branch else f"\n{kind}"
        ax.text(
            mid_x,
            mid_y + 0.13,
            f"{branch['id']}{detail}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#34495e",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
            zorder=4,
        )

    for bus in network["buses"]:
        bus_id = str(bus["id"])
        x, y = positions[bus_id]
        is_slack = bool(bus.get("is_slack", False))
        devices = devices_by_bus.get(bus_id, [])
        loads = [device for device in devices if device["kind"] == "load"]
        ders = [device for device in devices if device["kind"] != "load"]
        fill = "#f4d03f" if is_slack else "#d6eaf8" if loads else "#f7f9f9"
        marker = "s" if is_slack else "o"
        ax.scatter(x, y, s=760 if is_slack else 620, marker=marker, c=fill, edgecolors="#1f2d3d", linewidths=1.2, zorder=3)
        node_type = "Source / slack" if is_slack else "Load bus" if loads else "Junction bus"
        label = f"{bus_id}\n{node_type}"
        if loads:
            label += "\n" + ", ".join(str(device["id"]) for device in loads)
        if ders:
            label += "\n" + ", ".join(
                f"{DEVICE_STYLE[device['kind']][0]}: {device['id']}"
                for device in ders
            )
        ax.annotate(label, (x, y), xytext=(0, -24), textcoords="offset points", ha="center", va="top", fontsize=7.5, zorder=5)
        for index, device in enumerate(ders):
            _, color = DEVICE_STYLE[device["kind"]]
            ax.scatter(x - 0.22 + 0.15 * index, y + 0.18, s=65, marker="D", c=color, edgecolors="#34495e", linewidths=0.5, zorder=4)

    legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#f4d03f", markeredgecolor="#1f2d3d", label="Source / slack bus", markersize=10),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d6eaf8", markeredgecolor="#1f2d3d", label="Load bus", markersize=10),
        Line2D([0], [0], color="#566573", label="Line"),
        Line2D([0], [0], color="#9b59b6", linestyle="--", label="Transformer / regulator"),
        Line2D([0], [0], color="#c0392b", linestyle=":", label="Switch"),
    ] + [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=color, markeredgecolor="#34495e", label=label, markersize=7)
        for label, color in DEVICE_STYLE.values()
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8, frameon=True, ncol=2)
    ax.set_title("Generated feeder topology (node type, connected devices, and branches)", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
