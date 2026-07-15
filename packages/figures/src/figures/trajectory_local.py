r"""Figures for the local class-profile displacement (the score-invariance recast, plan 7e).

An ``invariance-trajectory`` run reads how each pooled class centroid moves as a smooth function
of the axis, under frozen responsibilities, with a family-clustered bootstrap tube. These figures
are a view of that full-dimensional effect, not its authority: each carries the in-plane capture
fraction, so a class whose drift is mostly out of the discriminant plane is flagged rather than
flattered by the picture.

Three figures:

- :func:`plane_figure` draws all four classes in one discriminant plane: the pooled anchors, each
  class's local trajectory with a time arrow, the centroid bootstrap tube, and the capture
  fraction;
- :func:`panels_figure` gives one panel per class, the trajectory and its tube over the faint
  within-class member ellipse, which answer different questions (where the centroid sits versus
  where the members spread);
- :func:`specificity_figure` is the small-multiple that reads the separation-scaled endpoint
  displacement of the timing axes against the control panel (household income, area deprivation, a
  random ordering), so the timing effect is shown to be larger than a control rather than merely
  non-zero;
- :func:`directional_figure` (DIREC, plan 12b) draws each class's one-dimensional signed
  trajectory, the projection onto its net direction, with the clustered-bootstrap band and the
  descriptive single-break location, so a monotone trend and a boundary discontinuity are visible;
- :func:`referent_figure` (ATTR-REF, era only) draws, per class, the size-fair current-state and
  retrospective root-mean-square drift intensity with the per-instrument underlay, so the
  measurement-timing signature (current-dominant) and the diagnosed-population signature
  (retrospective-dominant) are read off the two-way split.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, FancyArrowPatch

from figures import style
from figures.trajectory import (
    _COVERAGE_LEVELS,
    _coverage_alpha,
    _coverage_ellipse,
    _coverage_radius,
)

_LETTERS = ("A", "B", "C", "D")
_NICE_AXIS = {"age_at_diagnosis": "age at diagnosis", "era": "diagnostic era"}
_NICE_REFERENT = {"current_state": "current state", "retrospective": "retrospective"}
_NICE_INSTRUMENT = {
    "rbsr": "RBS-R",
    "cbcl_6_18": "CBCL 6-18",
    "scq": "SCQ (Lifetime)",
    "background_history_child": "milestones / history",
    "background_history_sibling": "sibling history",
}
_NICE_CONTROL = {
    "household_income": "household income",
    "area_deprivation": "area deprivation",
    "sex": "sex",
    "random": "random order",
    "era": "diagnostic era",
    "age_at_diagnosis": "age at diagnosis",
}
# A class whose in-plane capture falls below this is flagged: the 2D picture understates it.
_LOW_CAPTURE = 0.5


def _tube_ellipse(centre: np.ndarray, half: np.ndarray, colour: str, alpha: float) -> Ellipse:
    """Return a light axis-aligned ellipse standing for the centroid tube at one focal point."""
    return Ellipse(
        tuple(centre),
        float(2.0 * max(half[0], 1e-6)),
        float(2.0 * max(half[1], 1e-6)),
        facecolor=colour,
        edgecolor="none",
        alpha=alpha,
        zorder=2.0,
    )


def _class_order(anchors: pd.DataFrame) -> list[int]:
    """Return the class ids in a stable order."""
    return sorted(int(c) for c in anchors["ref_class"])


def plane_figure(
    plane: pd.DataFrame,
    capture: pd.DataFrame,
    meta: dict,
    *,
    arrow: bool = True,
    width_in: float | None = None,
    height_in: float | None = None,
    brief: bool = False,
) -> Figure:
    """Build the combined four-class discriminant-plane trajectory figure.

    Each class is a run of focal-point dots coloured low-to-high along the ordering, wrapped in the
    translucent bootstrap tube, with a black-edged marker at the pooled anchor and the in-plane
    capture fraction in the label. A colourbar keys the dot colour to the ordering.

    Parameters
    ----------
    plane : pandas.DataFrame
        The ``trajectory_<axis>`` table: anchor rows (with the member covariance) and per-focal
        centroid rows (with the bootstrap tube box ``ld1_lo``, ``ld1_hi``, ``ld2_lo``, ``ld2_hi``).
    capture : pandas.DataFrame
        The ``capture_<axis>`` table, carrying the per-class in-plane capture fraction.
    meta : dict
        The run's manifest metrics, carrying ``axis``.
    arrow : bool, optional
        When true (the default) draw a net-drift arrow from the early to the late focal points of
        each class; set it false for the dots and tube alone.
    width_in, height_in : float, optional
        The figure size in inches; sensible defaults are used when omitted.
    brief : bool, optional
        When true, drop the panel letter for a document that supplies its own caption.

    Returns
    -------
    matplotlib.figure.Figure
        The single-panel figure.
    """
    anchors = plane[plane["kind"] == "anchor"].set_index("ref_class")
    focal = plane[plane["kind"] == "focal"]
    classes = _class_order(plane[plane["kind"] == "anchor"])
    colours = {c: style.PALETTE[i % len(style.PALETTE)] for i, c in enumerate(classes)}
    names = {c: str(anchors.loc[c, "class_name"]) for c in classes}
    capture_of = dict(
        zip(capture["ref_class"].astype(int), capture["capture"].astype(float), strict=True)
    )
    cmap = plt.get_cmap("viridis")
    # Colour the focal dots by the ordering variable's own value (age at diagnosis in years,
    # diagnostic era as a calendar year), so the colourbar reads in real units, not a rank.
    positions = focal["position"].astype(float)
    norm = Normalize(float(positions.min()), float(positions.max()))
    nice = _NICE_AXIS.get(str(meta.get("axis")), str(meta.get("axis")))

    width = width_in if width_in is not None else 7.6
    height = height_in if height_in is not None else 6.4
    # In the narrow brief column a class sitting near the right edge would have its label run into
    # the colourbar, so labels past this cut are anchored on their right and grow leftward instead.
    anchor_ld1 = anchors["ld1"].astype(float)
    right_zone = float(anchor_ld1.min()) + 0.6 * (float(anchor_ld1.max()) - float(anchor_ld1.min()))

    with style.house_style():
        fig, ax = plt.subplots(figsize=(width, height))
        for c in classes:
            path = focal[focal["ref_class"] == c].sort_values("focal_index")
            pts = path[["ld1", "ld2"]].to_numpy(dtype=float)
            shade = path["position"].to_numpy(dtype=float)
            half1 = (path["ld1_hi"].to_numpy() - path["ld1_lo"].to_numpy()) / 2.0
            half2 = (path["ld2_hi"].to_numpy() - path["ld2_lo"].to_numpy()) / 2.0
            for k in range(pts.shape[0]):
                ax.add_patch(
                    _tube_ellipse(pts[k], np.array([half1[k], half2[k]]), colours[c], 0.16)
                )
            ax.plot(pts[:, 0], pts[:, 1], color=colours[c], lw=0.8, alpha=0.5, zorder=3)
            ax.scatter(
                pts[:, 0], pts[:, 1], s=16, c=shade, cmap=cmap, norm=norm, zorder=4, linewidths=0
            )
            if arrow:
                third = max(2, pts.shape[0] // 3)
                ax.add_patch(
                    FancyArrowPatch(
                        pts[:third].mean(axis=0),
                        pts[-third:].mean(axis=0),
                        arrowstyle="-|>",
                        mutation_scale=16,
                        color=colours[c],
                        lw=2.0,
                        alpha=0.9,
                        zorder=5,
                    )
                )
            anchor = anchors.loc[c, ["ld1", "ld2"]].to_numpy(dtype=float)
            ax.scatter(*anchor, s=42, color=colours[c], edgecolor="black", lw=0.6, zorder=6)
            flag = " *" if capture_of.get(c, 1.0) < _LOW_CAPTURE else ""
            # The brief packs this figure into a half-column, so the labels drop the capture
            # parenthetical (the caption carries it) and sit in a smaller font.
            label = names[c].split()[0]
            if not brief:
                label += f" (capture {capture_of.get(c, float('nan')):.2f}{flag})"
            flip = brief and anchor[0] > right_zone
            ax.annotate(
                label,
                anchor,
                textcoords="offset points",
                xytext=(-4, 4) if flip else ((4, 4) if brief else (6, 6)),
                ha="right" if flip else "left",
                fontsize=6 if brief else 7,
                color=colours[c],
                zorder=7,
            )
        ax.set_xlabel("Linear discriminant 1")
        ax.set_ylabel("Linear discriminant 2")
        smap = ScalarMappable(norm=norm, cmap=cmap)
        smap.set_array([])
        bar = fig.colorbar(smap, ax=ax, fraction=0.045, pad=0.02)
        bar.set_label(nice if brief else f"Focal point ({nice})", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        if not brief:
            style.panel_title(ax, "A", f"Local class trajectories along {nice}")
        low = [names[c] for c in classes if capture_of.get(c, 1.0) < _LOW_CAPTURE]
        if low and not brief:
            ax.text(
                0.5,
                -0.13,
                "* drift is mostly out of plane; the full-dimensional magnitude is the authority",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=6.6,
                color="#555",
            )
    return fig


_AXIS_STYLE: dict[str, str | tuple[int, tuple[int, int]]] = {
    "age_at_diagnosis": "-",
    "era": (0, (5, 2)),
}


def _focal_endpoints(path: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a focal path's points, its start, and its end (mean of the outer thirds)."""
    pts = path.sort_values("focal_index")[["ld1", "ld2"]].to_numpy(dtype=float)
    third = max(2, pts.shape[0] // 3)
    return pts, pts[:third].mean(axis=0), pts[-third:].mean(axis=0)


def plane_overlay_figure(
    planes: dict[str, pd.DataFrame],
    captures: dict[str, pd.DataFrame],
    *,
    width_in: float | None = None,
    height_in: float | None = None,
    brief: bool = False,
) -> Figure:
    """Overlay the local trajectories of two timing axes in the one shared discriminant plane.

    Both axes are read against the same pooled fit, so the class anchors and the discriminant
    basis are common: each class grows two trajectories from its anchor, one per axis, drawn in
    the class colour and told apart by line style (age at diagnosis solid, diagnostic era dashed).
    The colour-by-focal-point scale of :func:`plane_figure` is dropped, since one scale cannot
    serve two axes; a small start dot and the arrowhead carry the low-to-high direction instead.

    Parameters
    ----------
    planes : dict of str to pandas.DataFrame
        The ``trajectory_<axis>`` table per axis name (``"age_at_diagnosis"``, ``"era"``); the
        anchors are taken from the first and the others must share them.
    captures : dict of str to pandas.DataFrame
        The ``capture_<axis>`` table per axis name, for the in-plane capture note.
    width_in : float, optional
        The figure width in inches; the height follows the default aspect. Set this to the
        document text width so the input is placed at natural size.
    height_in : float, optional
        The figure height in inches; when omitted it follows the default aspect from the width.
    brief : bool, optional
        When true, drop the panel letter and the in-axes capture note for a document that
        supplies its own caption.

    Returns
    -------
    matplotlib.figure.Figure
        The single-panel overlay figure.
    """
    axes_order = [a for a in ("age_at_diagnosis", "era") if a in planes]
    base = planes[axes_order[0]]
    anchors = base[base["kind"] == "anchor"].set_index("ref_class")
    classes = _class_order(base[base["kind"] == "anchor"])
    colours = {c: style.PALETTE[i % len(style.PALETTE)] for i, c in enumerate(classes)}
    names = {c: str(anchors.loc[c, "class_name"]) for c in classes}
    capture_of = {
        axis: dict(zip(cap["ref_class"].astype(int), cap["capture"].astype(float), strict=True))
        for axis, cap in captures.items()
    }
    width = width_in if width_in is not None else 7.6
    height = height_in if height_in is not None else width * (6.4 / 7.6)

    with style.house_style():
        fig, ax = plt.subplots(figsize=(width, height))
        for c in classes:
            colour = colours[c]
            for axis in axes_order:
                plane = planes[axis]
                path = plane[(plane["kind"] == "focal") & (plane["ref_class"] == c)]
                _, start, end = _focal_endpoints(path)
                # A single straight line from the start of the ordering to its most recent value,
                # the arrowhead marking that recent end.
                ax.add_patch(
                    FancyArrowPatch(
                        (float(start[0]), float(start[1])),
                        (float(end[0]), float(end[1])),
                        arrowstyle="-|>",
                        mutation_scale=12,
                        color=colour,
                        lw=1.2,
                        ls=_AXIS_STYLE[axis],
                        alpha=0.9,
                        zorder=4,
                    )
                )
            anchor = anchors.loc[c, ["ld1", "ld2"]].to_numpy(dtype=float)
            # An x marks the pooled centroid location for the class.
            ax.scatter(*anchor, marker="x", s=34, color=colour, linewidths=1.3, zorder=6)
            ax.annotate(
                names[c].split()[0],
                anchor,
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
                color=colour,
                zorder=7,
            )
        ax.set_xlabel("Linear discriminant 1")
        ax.set_ylabel("Linear discriminant 2")
        style.brief_axis_style(ax)
        # The line-style legend only earns its place when two axes are overlaid; a single-axis
        # plot reads from the arrows alone, so it is left off.
        if len(axes_order) > 1:
            legend_handles = [
                plt.Line2D(
                    [], [], color="#444", lw=1.8, ls=_AXIS_STYLE[axis], label=_NICE_AXIS[axis]
                )
                for axis in axes_order
            ]
            ax.legend(handles=legend_handles, loc="upper left", fontsize=7)
        # The docs figure carries its own panel title and the capture caveat in the axes; the brief
        # supplies both through its LaTeX caption, so the small figure stays uncluttered.
        if not brief:
            style.panel_title(ax, "A", "Local class trajectories along both timing axes")
            worst = min(
                (v for table in capture_of.values() for v in table.values()), default=float("nan")
            )
            ax.text(
                0.5,
                -0.13,
                "arrows show the in-plane drift; most of it is out of plane "
                f"(capture down to {worst:.2f}), so the magnitude is read off the "
                "specificity panel",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=6.4,
                color="#555",
            )
    return fig


def panels_figure(plane: pd.DataFrame, capture: pd.DataFrame, meta: dict) -> Figure:
    """Build the per-class panels: the local trajectory and tube over the member ellipse.

    Parameters
    ----------
    plane : pandas.DataFrame
        The ``trajectory_<axis>`` table (anchors with the member covariance and per-focal
        centroids with the tube box).
    capture : pandas.DataFrame
        The ``capture_<axis>`` table.
    meta : dict
        The run's manifest metrics, carrying ``axis``.

    Returns
    -------
    matplotlib.figure.Figure
        The 2 by 2 figure, one panel per class.
    """
    anchors = plane[plane["kind"] == "anchor"].set_index("ref_class")
    focal = plane[plane["kind"] == "focal"]
    classes = _class_order(plane[plane["kind"] == "anchor"])
    colours = {c: style.PALETTE[i % len(style.PALETTE)] for i, c in enumerate(classes)}
    names = {c: str(anchors.loc[c, "class_name"]) for c in classes}
    capture_of = dict(
        zip(capture["ref_class"].astype(int), capture["capture"].astype(float), strict=True)
    )
    n_focal = int(focal["focal_index"].max()) + 1
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0, max(n_focal - 1, 1))
    nice = _NICE_AXIS.get(str(meta.get("axis")), str(meta.get("axis")))

    def _cov(c: int) -> np.ndarray:
        row = anchors.loc[c]
        return np.array([[row["cov11"], row["cov12"]], [row["cov12"], row["cov22"]]], dtype=float)

    with style.house_style():
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.4), sharex=True, sharey=True)
        for panel, (letter, ax) in enumerate(zip(_LETTERS, axes.flat, strict=False)):
            if panel >= len(classes):
                ax.axis("off")
                continue
            c = classes[panel]
            colour = colours[c]
            anchor = anchors.loc[c, ["ld1", "ld2"]].to_numpy(dtype=float)
            for coverage in _COVERAGE_LEVELS:
                ax.add_patch(
                    _coverage_ellipse(
                        anchor,
                        _cov(c),
                        _coverage_radius(coverage),
                        "#6f6f6f",
                        lw=0.9,
                        zorder=1.1,
                        alpha=_coverage_alpha(coverage),
                    )
                )
            path = focal[focal["ref_class"] == c].sort_values("focal_index")
            pts = path[["ld1", "ld2"]].to_numpy(dtype=float)
            order = path["focal_index"].to_numpy()
            half1 = (path["ld1_hi"].to_numpy() - path["ld1_lo"].to_numpy()) / 2.0
            half2 = (path["ld2_hi"].to_numpy() - path["ld2_lo"].to_numpy()) / 2.0
            for k in range(pts.shape[0]):
                ax.add_patch(_tube_ellipse(pts[k], np.array([half1[k], half2[k]]), colour, 0.22))
            ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=0.9, alpha=0.6, zorder=3)
            ax.scatter(
                pts[:, 0], pts[:, 1], s=18, c=order, cmap=cmap, norm=norm, zorder=4, linewidths=0
            )
            ax.scatter(*anchor, s=34, color=colour, edgecolor="black", lw=0.5, zorder=5)
            style.panel_title(ax, letter, names[c])
            ax.text(
                0.98,
                0.03,
                f"capture {capture_of.get(c, float('nan')):.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.6,
                color="#555",
            )
        for ax in axes[1, :]:
            ax.set_xlabel("Linear discriminant 1")
        for ax in axes[:, 0]:
            ax.set_ylabel("Linear discriminant 2")
        smap = ScalarMappable(norm=norm, cmap=cmap)
        smap.set_array([])
        bar = fig.colorbar(smap, ax=axes, fraction=0.03, pad=0.02)
        bar.set_label(f"Focal point ({nice}, low to high)", fontsize=8)
        bar.set_ticks([0, max(n_focal - 1, 1)])
        bar.set_ticklabels(["low", "high"], fontsize=7)
    return fig


def specificity_figure(
    specificity: pd.DataFrame,
    meta: dict,
    *,
    width_in: float | None = None,
    height_in: float | None = None,
    brief: bool = False,
) -> Figure:
    """Build the specificity small-multiple: endpoint displacement by axis, per class.

    The separation-scaled endpoint magnitude of each axis, with the timing axes drawn against the
    control panel. A per-class dot sits on each axis's bar (the mean over classes), so the
    comparison is read as a magnitude ordering rather than a reject-or-not decision.

    Parameters
    ----------
    specificity : pandas.DataFrame
        The merged ``specificity`` rows, with ``axis_name``, ``ref_class``, ``class_name`` and
        ``endpoint_magnitude``. The timing axes and the control axes are pooled here.
    meta : dict
        Presentation metrics; ``timing_axes`` names the axes drawn as the effect (highlighted).
    width_in : float, optional
        The figure width in inches; the height follows the default aspect unless ``height_in`` is
        given. Set this to the document text width so the input is placed at natural size.
    height_in : float, optional
        The figure height in inches, overriding the default aspect. Use it to flatten the panel
        (the five long axis labels need close to the full text width, so height is the free knob).
    brief : bool, optional
        When true, drop the panel letter for a document that supplies its own caption.

    Returns
    -------
    matplotlib.figure.Figure
        The single-panel figure.
    """
    timing = set(meta.get("timing_axes", ["era", "age_at_diagnosis"]))
    order = ["era", "age_at_diagnosis", "area_deprivation", "household_income", "sex", "random"]
    present = [a for a in order if a in set(specificity["axis_name"])]
    means = specificity.groupby("axis_name")["endpoint_magnitude"].mean()
    width = width_in if width_in is not None else 7.4
    height = height_in if height_in is not None else width * (4.4 / 7.4)

    with style.house_style():
        fig, ax = plt.subplots(figsize=(width, height))
        for i, axis_name in enumerate(present):
            highlight = axis_name in timing
            ax.bar(
                i,
                float(means[axis_name]),
                width=0.62,
                color=style.PALETTE[3] if highlight else "#b8b8b8",
                edgecolor="black" if highlight else "none",
                linewidth=0.6 if highlight else 0.0,
                zorder=2,
            )
            rows = specificity[specificity["axis_name"] == axis_name]
            jitter = np.linspace(-0.14, 0.14, len(rows))
            ax.scatter(
                i + jitter,
                rows["endpoint_magnitude"].to_numpy(dtype=float),
                s=16,
                color="#333",
                zorder=3,
            )
        ax.set_xticks(range(len(present)))
        # At a single column the five labels are too wide to sit horizontally, so the brief rotates
        # them; the wider docs panel keeps them flat.
        labels = [_NICE_CONTROL.get(a, a) for a in present]
        if brief:
            ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
        else:
            ax.set_xticklabels(labels, fontsize=8)
        # A flattened panel (brief) has too little height for the label on one rotated line, so
        # it wraps to two; the full-height docs panel keeps it on one.
        ax.set_ylabel(
            "Endpoint displacement\n(separation units)"
            if brief
            else "Endpoint displacement (separation units)"
        )
        control_mean = float(
            specificity[~specificity["axis_name"].isin(timing)]["endpoint_magnitude"].mean()
        )
        ax.axhline(control_mean, color=style.REFERENCE_COLOUR, ls=":", lw=0.9, zorder=1)
        # The docs panel labels the line in the axes; the brief has no room, so its caption does.
        if not brief:
            ax.text(
                len(present) - 0.5,
                control_mean,
                " control mean",
                va="center",
                ha="left",
                fontsize=6.8,
                color=style.REFERENCE_COLOUR,
            )
        if not brief:
            style.panel_title(ax, "A", "Specificity: timing drift against the control panel")
        ax.margins(x=0.04)
    return fig


def _bar_group(ax, x: float, values: dict[int, float], colours: dict[int, str], classes: list[int]):
    """Draw one axis group as per-class bars with a rule at the across-class mean.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The subplot to draw into.
    x : float
        The group's centre on the x-axis.
    values : dict of int to float
        The endpoint displacement per class id for this axis.
    colours : dict of int to str
        The bar colour per class id.
    classes : list of int
        The class ids in draw order (left to right within the group).
    """
    width = 0.8 / len(classes)
    for j, c in enumerate(classes):
        offset = (j - (len(classes) - 1) / 2.0) * width
        ax.bar(x + offset, values[c], width=width, color=colours[c], zorder=2)
    mean = float(np.mean([values[c] for c in classes]))
    ax.plot([x - 0.44, x + 0.44], [mean, mean], color="black", lw=1.3, zorder=4)


def specificity_panels_figure(
    specificity: pd.DataFrame,
    meta: dict,
    *,
    width_in: float | None = None,
    height_in: float | None = None,
) -> Figure:
    """Build the two-panel specificity figure: per-class timing drift vs a random control.

    Each ordering axis is a group of four bars, one per class in a consistent colour (legend), so
    the reader sees which class moves. A black rule across each group marks the across-class mean,
    the summary the single bar of :func:`specificity_figure` used to carry. Panel A holds the timing
    axes and panel B a random-order control; they share the y-axis and a dotted line at the control
    mean, so the timing bars clearing that line, and the random control sitting on it, is the read.

    Parameters
    ----------
    specificity : pandas.DataFrame
        The merged ``specificity`` rows (``axis_name``, ``ref_class``, ``class_name``,
        ``endpoint_magnitude``), timing and control axes pooled.
    meta : dict
        Presentation metrics; ``timing_axes`` names the axes shown in panel A.
    width_in, height_in : float, optional
        The figure size in inches; sensible defaults are used when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    timing = [a for a in ("era", "age_at_diagnosis") if a in set(specificity["axis_name"])]
    # The only negative control shown is a random re-ordering. A socioeconomic ordering (household
    # income, area deprivation) could shift the phenotype profile in its own right, so it is not a
    # clean null; the random order is the one baseline guaranteed to carry no real signal.
    controls = [a for a in ("random",) if a in set(specificity["axis_name"])]
    classes = sorted(int(c) for c in specificity["ref_class"].unique())
    colours = {c: style.PALETTE[i % len(style.PALETTE)] for i, c in enumerate(classes)}
    labelled = specificity.drop_duplicates("ref_class")
    names = {
        int(c): str(n).split()[0]
        for c, n in zip(labelled["ref_class"], labelled["class_name"], strict=True)
    }
    control_rows = specificity[specificity["axis_name"].isin(controls)]
    control_mean = float(control_rows["endpoint_magnitude"].mean())

    def _values(axis_name: str) -> dict[int, float]:
        rows = specificity[specificity["axis_name"] == axis_name]
        return dict(
            zip(
                rows["ref_class"].astype(int),
                rows["endpoint_magnitude"].astype(float),
                strict=True,
            )
        )

    width = width_in if width_in is not None else 3.2
    height = height_in if height_in is not None else 2.8
    with style.house_style():
        fig, (ax_a, ax_b) = plt.subplots(
            1,
            2,
            figsize=(width, height),
            sharey=True,
            gridspec_kw={
                "width_ratios": [max(len(timing), 1), max(len(controls), 1)],
                "wspace": 0.1,
            },
        )
        for ax, axes_here, letter, label in (
            (ax_a, timing, "A", "timing"),
            (ax_b, controls, "B", "control"),
        ):
            for i, axis_name in enumerate(axes_here):
                _bar_group(ax, i, _values(axis_name), colours, classes)
            ax.axhline(control_mean, color=style.REFERENCE_COLOUR, ls=":", lw=1.0, zorder=1)
            ax.set_xticks(range(len(axes_here)))
            ax.set_xticklabels(
                [_NICE_CONTROL.get(a, a) for a in axes_here], fontsize=7, rotation=30, ha="right"
            )
            ax.margins(x=0.08)
            style.panel_title(ax, letter, label)
            style.brief_axis_style(ax, minor_x=False)
        ax_a.set_ylabel("Endpoint displacement\n(separation units)")
        ax_b.text(
            len(controls) - 0.5,
            control_mean,
            " random mean",
            va="bottom",
            ha="right",
            fontsize=6.2,
            color=style.REFERENCE_COLOUR,
        )
        handles = [
            plt.Line2D([], [], marker="s", ls="", markersize=6, color=colours[c], label=names[c])
            for c in classes
        ]
        # Reserve a band at the bottom for the rotated tick labels, then put the legend in its own
        # band beneath them, so the two no longer collide. The legend wraps to two rows so it does
        # not overrun the narrow figure width.
        fig.subplots_adjust(bottom=0.34, top=0.9)
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=2,
            fontsize=6.5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.12),
        )
    return fig


def directional_figure(signed: pd.DataFrame, directional: pd.DataFrame, meta: dict) -> Figure:
    """Build the DIREC figure: each class's one-dimensional signed trajectory along the axis.

    Each class is projected onto its own net direction, giving a signed trajectory $s_k(f)$ whose
    slope is the directional statistic. The clustered-bootstrap band is shaded, the horizontal
    zero line is the pooled centroid, and a marker sits at the descriptive single-break location.
    The legend carries each class's separation-scaled net trend with its interval and whether it
    is directional, so the picture and the test agree.

    Parameters
    ----------
    signed : pandas.DataFrame
        The ``signed_trajectory_<axis>`` table (``ref_class``, ``position``, ``signed``,
        ``band_lo``, ``band_hi``).
    directional : pandas.DataFrame
        The per-class ``directional_<axis>`` summary (``net_trend``, its interval, ``reject``,
        ``break_position``).
    meta : dict
        The run's manifest metrics, carrying ``axis``.

    Returns
    -------
    matplotlib.figure.Figure
        The single-panel figure.
    """
    classes = sorted(int(c) for c in signed["ref_class"].unique())
    colours = {c: style.PALETTE[i % len(style.PALETTE)] for i, c in enumerate(classes)}
    summary = directional.set_index("ref_class")
    nice = _NICE_AXIS.get(str(meta.get("axis")), str(meta.get("axis")))

    with style.house_style():
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        ax.axhline(0.0, color=style.REFERENCE_COLOUR, ls="-", lw=0.8, zorder=1)
        for c in classes:
            path = signed[signed["ref_class"] == c].sort_values("position")
            pos = path["position"].to_numpy(dtype=float)
            s = path["signed"].to_numpy(dtype=float)
            colour = colours[c]
            row = summary.loc[c]
            ax.fill_between(
                pos,
                path["band_lo"].to_numpy(dtype=float),
                path["band_hi"].to_numpy(dtype=float),
                color=colour,
                alpha=0.14,
                lw=0,
                zorder=2,
            )
            name = str(row["class_name"]).split()[0]
            verdict = "directional" if bool(row["reject"]) else "ns"
            label = (
                f"{name}: net {row['net_trend']:+.1f} "
                f"[{row['net_trend_lo']:+.1f}, {row['net_trend_hi']:+.1f}], {verdict}"
            )
            ax.plot(pos, s, color=colour, lw=1.6, zorder=4, label=label)
            brk = float(row["break_position"])
            if np.isfinite(brk):
                ax.axvline(brk, color=colour, ls=":", lw=0.9, alpha=0.7, zorder=3)
        ax.set_xlabel(f"{nice[0].upper()}{nice[1:]}")
        ax.set_ylabel("Signed displacement along net direction (standardised)")
        ax.legend(loc="best", fontsize=6.6)
        style.panel_title(ax, "A", f"Directional drift of each class along {nice}")
        ax.text(
            0.5,
            -0.15,
            "dotted line: descriptive single-break location (the bridge supLM confidence set "
            "saturates at full n)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.4,
            color="#555",
        )
    return fig


def referent_figure(grains: pd.DataFrame, contrast: pd.DataFrame, meta: dict) -> Figure:
    """Build the ATTR-REF figure: per-class current-versus-retrospective drift intensity.

    One panel per class. Two bars give the size-fair root-mean-square displacement intensity of the
    current-state referent (RBS-R, CBCL 6-18) and the retrospective referent (SCQ Lifetime,
    developmental milestones and history); the per-instrument intensities are overlaid as points,
    the transparent underlay showing which instrument carries each referent's drift. The panel
    title reads the current-minus-retrospective contrast with its clustered-bootstrap interval and
    the mechanism it implies: a current-dominant class carries the measurement-timing signature, a
    retrospective-dominant class the diagnosed-population signature.

    Parameters
    ----------
    grains : pandas.DataFrame
        The ``referent_<axis>`` table (per class per grain: ``grain_kind``, ``grain``,
        ``referent``, ``rms``, ``share``, ``n_features``).
    contrast : pandas.DataFrame
        The per-class ``referent_contrast_<axis>`` summary (``contrast``, ``ci_low``, ``ci_high``,
        ``reject``, ``mechanism``).
    meta : dict
        The run's manifest metrics, carrying ``axis``.

    Returns
    -------
    matplotlib.figure.Figure
        The 2 by 2 figure, one panel per class.
    """
    order = ("current_state", "retrospective")
    colours = {"current_state": style.PALETTE[3], "retrospective": style.PALETTE[0]}
    summary = contrast.set_index("ref_class")
    classes = sorted(int(c) for c in contrast["ref_class"])
    referent_rows = grains[grains["grain_kind"] == "referent"]
    instrument_rows = grains[grains["grain_kind"] == "instrument"]
    nice = _NICE_AXIS.get(str(meta.get("axis")), str(meta.get("axis")))
    rms_ceiling = float(grains["rms"].max()) if len(grains) else 1.0

    with style.house_style():
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6), sharey=True)
        for panel, (letter, ax) in enumerate(zip(_LETTERS, axes.flat, strict=False)):
            if panel >= len(classes):
                ax.axis("off")
                continue
            c = classes[panel]
            ref_c = referent_rows[referent_rows["ref_class"] == c].set_index("grain")
            for i, referent in enumerate(order):
                if referent not in ref_c.index:
                    continue
                ax.bar(
                    i,
                    float(ref_c.loc[referent, "rms"]),
                    width=0.6,
                    color=colours[referent],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=2,
                )
                inst = instrument_rows[
                    (instrument_rows["ref_class"] == c) & (instrument_rows["referent"] == referent)
                ]
                jitter = np.linspace(-0.16, 0.16, len(inst)) if len(inst) else np.array([])
                ax.scatter(
                    i + jitter,
                    inst["rms"].to_numpy(dtype=float),
                    s=22,
                    color="#333",
                    zorder=3,
                )
                for xj, (_, row) in zip(jitter, inst.iterrows(), strict=False):
                    ax.annotate(
                        _NICE_INSTRUMENT.get(str(row["grain"]), str(row["grain"])),
                        (i + xj, float(row["rms"])),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha="center",
                        fontsize=5.6,
                        color="#333",
                        rotation=90,
                    )
            row = summary.loc[c]
            mechanism = str(row["mechanism"])
            star = " *" if bool(row["reject"]) else ""
            name = str(row["class_name"]).split()[0]
            style.panel_title(ax, letter, name)
            ax.text(
                0.5,
                0.97,
                f"current - retrospective {row['contrast']:+.2f} "
                f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}], {mechanism}{star}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=6.6,
                color=colours["current_state"]
                if mechanism == "timing"
                else colours["retrospective"],
            )
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([_NICE_REFERENT[r] for r in order], fontsize=8)
            ax.set_ylim(0.0, rms_ceiling * 1.5)
        for ax in axes[:, 0]:
            ax.set_ylabel("Per-feature RMS displacement (standardised)")
        fig.suptitle(f"Referent decomposition of the {nice} drift", fontsize=11, y=0.98)
        fig.text(
            0.5,
            0.005,
            "bars: size-fair per-referent RMS; points: per-instrument RMS underlay; "
            "* current-minus-retrospective contrast rejects (BH q=0.05)",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color="#555",
        )
    return fig
