"""Wykres Gantta najlepszego grafiku — statyczny (matplotlib) i interaktywny (plotly)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from amhe.model import labor_law as law  # noqa: E402
from amhe.model.schedule import ProblemInstance, Schedule  # noqa: E402

#: kolory zmian wg pory dnia
_DAY_COLORS = {
    "noc": "#34495e",
    "rano": "#f1c40f",
    "dzien": "#2ecc71",
    "wieczor": "#e67e22",
}


def _slot_period(start_slot: int) -> str:
    """Klasyfikuje zmiane wg slotu poczatku na pore dnia (do kolorowania)."""
    h = start_slot / law.SLOTS_PER_HOUR
    if h < 6 or h >= 22:
        return "noc"
    if h < 12:
        return "rano"
    if h < 17:
        return "dzien"
    return "wieczor"


def gantt_matplotlib(instance: ProblemInstance, schedule: Schedule, path,
                     title="Harmonogram tygodniowy"):
    """Statyczny wykres Gantta: os Y = pracownicy, os X = czas (dni x godziny)."""
    E, D = schedule.n_employees, schedule.n_days
    fig, ax = plt.subplots(figsize=(min(2 + D * 1.6, 16), 1 + E * 0.5))

    for e in range(E):
        for d in range(D):
            ln = int(schedule.length[e, d])
            if ln <= 0:
                continue
            st = int(schedule.start[e, d])
            x = d * 24 + st / law.SLOTS_PER_HOUR
            w = ln / law.SLOTS_PER_HOUR
            color = _DAY_COLORS[_slot_period(st)]
            ax.barh(e, w, left=x, height=0.6, color=color, edgecolor="black",
                    linewidth=0.4)

    for d in range(1, D):
        ax.axvline(d * 24, color="gray", linestyle=":", linewidth=0.6)

    ax.set_yticks(range(E))
    ax.set_yticklabels([emp.name for emp in instance.employees])
    ax.set_xticks([d * 24 + 12 for d in range(D)])
    ax.set_xticklabels([f"dz {d}" for d in range(D)])
    ax.set_xlim(0, D * 24)
    ax.set_xlabel("czas (kolejne dni)")
    ax.set_title(title)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in _DAY_COLORS.items()],
              loc="upper right", ncol=4, fontsize=8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def gantt_plotly(instance: ProblemInstance, schedule: Schedule, path,
                 title="Harmonogram tygodniowy (interaktywny)"):
    """Interaktywny wykres Gantta zapisany do pliku HTML (plotly)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for e in range(schedule.n_employees):
        name = instance.employees[e].name
        for d in range(schedule.n_days):
            ln = int(schedule.length[e, d])
            if ln <= 0:
                continue
            st = int(schedule.start[e, d])
            x0 = d * 24 + st / law.SLOTS_PER_HOUR
            x1 = x0 + ln / law.SLOTS_PER_HOUR
            period = _slot_period(st)
            fig.add_trace(go.Bar(
                x=[x1 - x0], base=[x0], y=[name], orientation="h",
                marker_color=_DAY_COLORS[period],
                hovertemplate=(f"{name}<br>dzien {d}<br>"
                               f"{x0 % 24:.1f}-{x1 % 24:.1f} h<br>{period}<extra></extra>"),
                showlegend=False,
            ))
    fig.update_layout(
        title=title, barmode="overlay",
        xaxis_title="czas (kolejne dni)", yaxis_title="pracownik",
        xaxis=dict(tickvals=[d * 24 + 12 for d in range(schedule.n_days)],
                   ticktext=[f"dz {d}" for d in range(schedule.n_days)]),
        height=200 + 30 * schedule.n_employees, template="plotly_white",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path.with_suffix(".html")))
