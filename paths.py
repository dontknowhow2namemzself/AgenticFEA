"""Where one pipeline run keeps its files: the per-run working directory.

Every stage used to define its own `out/...` constants, so every run wrote into
the same directory. That was harmless for one person at one terminal, but two
runs at once -- two browser tabs on the same dashboard, or a double-clicked
button -- overwrite each other's mesh, deck and results without
any error, and a viewer is shown somebody else's numbers.

`RunPaths` is the one place that knows every filename a run produces. The
orchestrator creates one per run and hands it down; each stage reads and writes
only through it. One object rather than one argument per file, so a stage can
never be handed the mesh of one run together with the results of another --
which is exactly the class of bug this replaces.

The default is still `out/`, so every documented command keeps working as it
did. Isolation is opt-in on the command line (`--workdir`) and always on in the
dashboard (`RunPaths.new_run()`).

The filenames themselves are unchanged (`beam.*`, plate plan D8): this module
is a refactor, not a rename. Now that the names live in one place, a later
rename is a one-line change here.
"""

from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "out"
RUNS_ROOT = DEFAULT_ROOT / "runs"

JOB = "beam"   # CalculiX job basename: ccx -i <job> reads <job>.inp, writes <job>.frd, ...

# Plate renders (top view): key -> filename. The cantilever's are below.
PLATE_PNG_NAMES = {
    "stress_xx": "stress_xx.png",
    "stress_yy": "stress_yy.png",
    "stress_xy": "stress_xy.png",
    "stress_vm": "stress_vm.png",
}
BEAM_PNG_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class RunPaths:
    """Every file of one run, rooted at one directory."""

    root: Path

    # ---- construction -------------------------------------------------
    @classmethod
    def default(cls) -> "RunPaths":
        """The shared `out/` directory every CLI command used before this."""
        return cls(DEFAULT_ROOT)

    @classmethod
    def at(cls, workdir) -> "RunPaths":
        """An explicit directory, or the default when none is given."""
        return cls(Path(workdir).expanduser().resolve()) if workdir else cls.default()

    @classmethod
    def new_run(cls, base: Path = RUNS_ROOT) -> "RunPaths":
        """A fresh, unique directory for one run: <base>/<YYYYmmdd-HHMMSS>-<6 hex>.

        Created with exist_ok=False, so two runs started in the same second can
        never share a directory -- the random suffix keeps them apart, and a
        collision would be an error rather than a silent overwrite.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        root = Path(base) / f"{stamp}-{secrets.token_hex(3)}"
        root.mkdir(parents=True, exist_ok=False)
        return cls(root)

    def ensure(self) -> "RunPaths":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    # ---- solver files ---------------------------------------------------
    @property
    def job(self) -> Path:
        return self.root / JOB

    @property
    def msh(self) -> Path:
        return self.root / f"{JOB}.msh"

    @property
    def inp(self) -> Path:
        return self.root / f"{JOB}.inp"

    @property
    def frd(self) -> Path:
        return self.root / f"{JOB}.frd"

    @property
    def vtu(self) -> Path:
        return self.root / f"{JOB}.vtu"

    @property
    def ccx_log(self) -> Path:
        return self.root / "ccx.log"

    # ---- reports and logs -----------------------------------------------
    @property
    def physics_report(self) -> Path:
        return self.root / "physics_report.txt"

    @property
    def verification_report(self) -> Path:
        return self.root / "verification_report.txt"

    @property
    def diagnostic_report(self) -> Path:
        return self.root / "diagnostic_report.txt"

    def stage_log(self, stage: str) -> Path:
        return self.root / f"stage_{stage}.log"

    @property
    def live_run_json(self) -> Path:
        return self.root / "live_run.json"

    # ---- renders --------------------------------------------------------
    @property
    def mesh_png(self) -> Path:
        return self.root / "mesh.png"

    @property
    def displacement_png(self) -> Path:
        return self.root / "displacement.png"

    def beam_stress_png(self, axis: str) -> Path:
        """Cantilever von Mises render, named by the axis the camera looks along."""
        if axis not in BEAM_PNG_AXES:
            raise ValueError(f"axis must be one of {BEAM_PNG_AXES}, got {axis!r}")
        return self.root / f"stress_{axis}.png"

    def plate_png(self, key: str) -> Path:
        return self.root / PLATE_PNG_NAMES[key]

    @property
    def viewer_npz(self) -> Path:
        """Surface, nodal results and element edges for the interactive 3D view."""
        return self.root / "viewer.npz"


def add_workdir_argument(parser: argparse.ArgumentParser) -> None:
    """The same --workdir option on every stage's command line."""
    parser.add_argument(
        "--workdir", default=None, metavar="DIR",
        help="run directory to read and write (default: out/)")


def from_args(args: argparse.Namespace) -> RunPaths:
    return RunPaths.at(getattr(args, "workdir", None)).ensure()
