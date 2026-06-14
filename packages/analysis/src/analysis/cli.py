"""analysis command-line interface (Typer).

The interface is designed as one subcommand per pipeline stage. Each stage will read named
inputs and write named outputs plus a manifest under ``artefacts/``, so a later run
recomputes only what changed. The planned order is: build the cohort matrix, fit and select
the reference model, check its stability, replicate it on the SSC, define the strata, re-fit
within them, and measure the drift against a permutation null.

The package is built stage by stage. Every subcommand below is a stub: it reports which
phase will add it and exits non-zero. The stubs are grouped under a "planned" panel in
``analysis --help`` so the status of each stage is visible without running it. As a stage is
written, its command moves out of that panel.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="analysis",
    help="Reproduce the Litman autism classes and test their stability across age at "
    "diagnosis and diagnostic era.",
    no_args_is_help=True,
    add_completion=False,
)

# Help panel for stages that are not implemented yet. A command leaves this panel when its
# stage is written, so `analysis --help` always shows which stages exist and which are planned.
_PLANNED = "Planned (not yet implemented)"


def _todo(stage: str, phase: int) -> None:
    """Report that ``stage`` is not implemented yet and exit non-zero."""
    msg = f"the {stage!r} stage is planned for phase {phase} and is not implemented yet"
    typer.echo(msg, err=True)
    raise typer.Exit(1)


@app.command(rich_help_panel=_PLANNED)
def cohort() -> None:
    """Build the harmonised proband-by-feature matrix and its typing manifest."""
    _todo("cohort", 1)


@app.command(rich_help_panel=_PLANNED)
def fit() -> None:
    """Fit the reference general finite mixture model and predict class labels."""
    _todo("fit", 1)


@app.command(rich_help_panel=_PLANNED)
def select() -> None:
    """Grid over the number of components and report the information criteria."""
    _todo("select", 1)


@app.command(rich_help_panel=_PLANNED)
def stability() -> None:
    """Summarise multi-initialisation and subsampling stability of the reference fit."""
    _todo("stability", 2)


@app.command(rich_help_panel=_PLANNED)
def replicate() -> None:
    """Project the reference model onto the SSC and correlate the category profiles."""
    _todo("replicate", 2)


@app.command(rich_help_panel=_PLANNED)
def strata() -> None:
    """Assign each proband to an age-at-diagnosis and a diagnostic-era stratum."""
    _todo("strata", 4)


@app.command(rich_help_panel=_PLANNED)
def stratify() -> None:
    """Re-estimate the model independently within each stratum of an axis."""
    _todo("stratify", 4)


@app.command(rich_help_panel=_PLANNED)
def drift() -> None:
    """Align stratum classes to the reference and measure drift against the null."""
    _todo("drift", 4)


@app.command(rich_help_panel=_PLANNED)
def sensitivity() -> None:
    """Re-fit under alternative feature sets and within cognitive-level strata."""
    _todo("sensitivity", 5)


@app.command(rich_help_panel=_PLANNED)
def report() -> None:
    """Assemble the non-disclosive tables and figures for the manuscript."""
    _todo("report", 7)


if __name__ == "__main__":
    app()
