#!/usr/bin/env python3
"""Technical report: a short PDF of one run, to hand over.

Written by deterministic code from the run record -- the record the dashboard
shows -- and the run's renders; no language model writes or edits it. It
states the load case (geometry, material, supports, load, mesh and every
assumption applied), the results, the verification against the reference, the
validity checks of the linear-elastic model, the time each stage took, and the
field plots. The same record always gives the same file.

Usage:  python report.py demo/cantilever_pass/run.json [-o report.pdf]
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import textwrap
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.lines import Line2D

from knowledge.reference_solutions import get_case

HERE = Path(__file__).resolve().parent

# A4 portrait, in inches.
W, H = 8.27, 11.69
LEFT, TOP, BOTTOM = 0.8, 0.75, 0.95
TEXT_W = W - 2 * LEFT
LABEL_W = 1.7                       # key column of the key-value rows
BODY, SMALL = 8.6, 7.2              # font sizes [pt]
INK, MUTED, RULE = "#1b1f24", "#5b6470", "#c9ced6"
STATUS_C = {"PASS": "#1a7f37", "OK": "#1a7f37", "FAIL": "#c62828", "EXCEEDED": "#a35c00",
            "WARNING": "#a35c00", "N/A": "#5b6470", "NOT CHECKED": "#5b6470"}

CHAIN = ("preprocess", "solve", "physics", "postprocess", "verify")   # the FE analysis proper

_PLACES = {"face_x_min": "end face x = 0", "face_x_max": "end face x = L",
           "face_y_min": "side face y = −b/2", "face_y_max": "side face y = +b/2",
           "face_z_min": "bottom face z = −h/2", "face_z_max": "top face z = +h/2",
           "edge_x_min": "left edge x = 0", "edge_x_max": "right edge x = a",
           "edge_y_min": "edge y = 0", "edge_y_max": "edge y = b", "hole_boundary": "hole edge"}
_DOFS = {1: "u_x", 2: "u_y", 3: "u_z"}
_ELEMENTS = {"C3D20R": "20-node hexahedra, reduced integration",
             "CPS8R": "8-node plane-stress quadrilaterals, reduced integration",
             "CPS6": "6-node plane-stress triangles"}
_VIEWS = {"x": "von Mises stress, end view", "y": "von Mises stress, side view",
          "z": "von Mises stress, top view", "vM": "von Mises stress",
          "σxx": "normal stress σxx", "σyy": "normal stress σyy", "σxy": "shear stress σxy",
          "disp": "displacement magnitude, deformed shape"}


def recorded_params(rec: dict) -> dict:
    """The validated parameter set a run was solved with (its Parameterize stage)."""
    text = next(a["text"] for s in rec["stages"] if s["id"] == "parameterize"
                for a in s.get("artifacts", []) if a.get("text", "").startswith("PARAMS ="))
    return ast.literal_eval(text.split("=", 1)[1])


def ident(rec: dict) -> str:
    """The run's name: its directory under out/runs/, or the demo it is."""
    return Path(rec.get("workdir") or rec.get("run_id", "run")).name


def file_name(rec: dict) -> str:
    return f"FEA-report-{ident(rec)}.pdf"


# ---------------------------------------------------------------- what is said

def _geometry(p: dict) -> str:
    g = p["geometry"]
    if g["type"] == "plate_with_hole":
        d = g["plate_with_hole"]
        return (f"plate with a hole, 2D plane stress: length a = {d['a']:g} mm (x, load direction), "
                f"width b = {d['b']:g} mm (y), thickness t = {d['t']:g} mm; hole radius "
                f"r = {d['r']:g} mm, centred at ({d['xc']:g}, {d['yc']:g}) mm")
    d = g["cantilever_beam"]
    return (f"cantilever beam, solid rectangular section: length L = {d['L']:g} mm (x), "
            f"width b = {d['b']:g} mm (y), height h = {d['h']:g} mm (z); "
            f"slenderness L/h = {d['L'] / d['h']:.3g}")


def _direction(v) -> str:
    axis = [i for i, c in enumerate(v) if c]
    if len(axis) == 1:
        return f"{'+' if v[axis[0]] > 0 else '−'}{'xyz'[axis[0]]}"
    return str(list(v))


def _supports(p: dict) -> list:
    out = []
    for f in p["bc"]["fixed"]:
        if isinstance(f, str):
            out.append(f"{_PLACES.get(f, f)}: fully fixed (all displacements zero)")
            continue
        where = _PLACES.get(f["face"], f["face"])
        if f.get("at") == "mid":
            where = f"the node nearest the middle of the {where}"
        out.append(f"{where}: " + ", ".join(f"{_DOFS.get(d, d)} = 0" for d in f["dofs"]))
    return out


def _loads(p: dict, pre: dict) -> list:
    out = []
    for ld in p["bc"]["loads"]:
        where = _PLACES.get(ld["face"], ld["face"])
        if p["geometry"]["type"] == "plate_with_hole":
            out.append(f"F = {ld['magnitude']:g} N in {_direction(ld['dir'])}, "
                       f"a uniform traction over the {where}")
        else:
            nodes = f" ({pre['load_nodes']} nodes)" if pre.get("load_nodes") else ""
            out.append(f"F = {ld['magnitude']:g} N in {_direction(ld['dir'])}, "
                       f"spread over the {where}{nodes}")
    return out


def _mesh(p: dict, pre: dict) -> str:
    kind = str(pre.get("type", ""))
    names = [n for n in _ELEMENTS if n in kind]
    what = "; ".join(f"{n}: {_ELEMENTS[n]}" for n in names)
    size = f"element size {p['mesh']['elem_size']:g} mm"
    if p["geometry"]["type"] == "plate_with_hole":
        size += " far from the hole, refined around it"
    # The plate's record already counts each type ("1313 x CPS8R"); the beam's names one.
    count = kind.replace(" x ", " × ") if kind[:1].isdigit() else f"{pre.get('elements')} × {kind}"
    parts = [count, f"{pre.get('nodes')} nodes", size]
    if pre.get("minSICN") is not None:
        parts.append(f"lowest element quality (min. SICN) {pre['minSICN']:.3f}")
    return " · ".join(parts) + (f"\n{what}" if what else "")


def _peak_note(is_plate: bool) -> str:
    if is_plate:
        return "at the hole edge, where it converges with mesh refinement"
    return ("at the clamped edge, where the FE stress is singular and rises with mesh refinement: "
            "not a design value (the yield check uses the nominal M c / I)")


# ---------------------------------------------------------------- page layout

def _lines(text: str, width_in: float, size: float) -> list:
    """Wrap to a width; continuation lines of a '• ' item are indented."""
    chars = max(20, int(width_in * 72 / (size * 0.57)))
    out = []
    for para in str(text).split("\n"):
        indent = "   " if para.startswith("• ") else ""
        out += textwrap.wrap(para, chars, subsequent_indent=indent) or [""]
    return out


class _Pages:
    """A4 pages filled from the top down; a block that does not fit starts a new page."""

    def __init__(self):
        self.figs = []
        self.new_page()

    def new_page(self):
        self.fig = Figure(figsize=(W, H))
        self.figs.append(self.fig)
        self.y = TOP

    def need(self, height: float):
        if self.y + height > H - BOTTOM:
            self.new_page()

    def put(self, x: float, s: str, size: float = BODY, dy: float = 0.0, **kw):
        """Text whose top sits at the cursor (+ dy), x inches from the left edge."""
        kw.setdefault("color", INK)
        self.fig.text(x / W, 1 - (self.y + dy) / H, s, fontsize=size, va="top", **kw)

    def rule(self, gap: float = 0.09):
        self.y += gap
        self.fig.add_artist(Line2D([LEFT / W, 1 - LEFT / W], [1 - self.y / H] * 2,
                                   color=RULE, lw=0.6))
        self.y += gap

    def heading(self, title: str, keep: float = 1.2):
        """A section title, kept on one page with the first `keep` inches below it."""
        self.need(0.45 + keep)
        self.y += 0.16
        self.put(LEFT, title, size=10.5, weight="bold")
        self.y += 0.27

    def row(self, label: str, value: str, color: str = INK):
        step = BODY * 1.45 / 72
        lines = _lines(value, TEXT_W - LABEL_W, BODY)
        self.need(len(lines) * step)
        self.put(LEFT, label, color=MUTED)
        for i, line in enumerate(lines):
            self.put(LEFT + LABEL_W, line, dy=i * step, color=color)
        self.y += len(lines) * step + 0.04

    def note(self, text: str):
        step = SMALL * 1.45 / 72
        lines = _lines(text, TEXT_W, SMALL)
        self.need(len(lines) * step)
        for i, line in enumerate(lines):
            self.put(LEFT, line, size=SMALL, dy=i * step, color=MUTED)
        self.y += len(lines) * step + 0.04


# ---------------------------------------------------------------- the report

def build(rec: dict) -> bytes:
    """The PDF of one run record, as bytes."""
    p = recorded_params(rec)
    is_plate = p["geometry"]["type"] == "plate_with_hole"
    st = {s["id"]: s for s in rec["stages"]}
    pre = st.get("preprocess", {}).get("data", {})
    post = st.get("postprocess", {}).get("data", {})
    v = st.get("verify", {}).get("data", {})
    checks = next((a["items"] for a in st.get("verify", {}).get("artifacts", [])
                   if a.get("kind") == "checks"), [])
    meta = rec.get("meta", {})
    title = rec.get("title") or rec.get("description", "")[:60]

    pg = _Pages()
    # -- header
    pg.put(LEFT, "T E C H N I C A L   R E P O R T   ·   L I N E A R   S T A T I C   F E   A N A L Y S I S",
           size=SMALL, color=MUTED)
    pg.y += 0.24
    pg.put(LEFT, title, size=15, weight="bold")
    pg.y += 0.36
    when = (rec["started"].replace("T", " ")[:16] + " UTC") if rec.get("started") else "recorded example"
    pg.put(LEFT, f"run {ident(rec)} · {when} · {meta.get('solver', '')} · {meta.get('host', '')}",
           size=SMALL, color=MUTED)
    pg.y += 0.24
    verdict = rec.get("verdict", "—")
    pg.put(LEFT, "verdict", color=MUTED)
    pg.put(LEFT + LABEL_W, verdict, weight="bold", color=STATUS_C.get(verdict, INK))
    if rec.get("warnings"):
        pg.put(LEFT + LABEL_W + 0.55, "⚠ outside the linear-elastic model's validity: "
               + "; ".join(w.split(":")[0] for w in rec["warnings"]), color=STATUS_C["WARNING"])
    pg.y += 0.12
    pg.rule()

    # -- 1 load case
    pg.heading("1   Load case")
    pg.row("problem statement", f"“{rec.get('description', '')}”")
    a = p.get("analysis", {})
    pg.row("analysis", f"{a.get('type', 'static')}, "
           + ("nonlinear" if a.get("nonlinear") else "linear: linear-elastic material, small displacements"))
    pg.row("geometry", _geometry(p))
    m = p["material"]
    pg.row("material", f"{m['name']}: E = {m['E']:g} MPa, ν = {m['nu']:g}, ρ = {m['rho']:g} t/mm³")
    pg.row("yield strength", v.get("yield") or "—")
    pg.row("supports", "\n".join(_supports(p)))
    pg.row("load", "\n".join(_loads(p, pre)))
    pg.row("mesh", _mesh(p, pre))
    pg.row("units", "SI-mm: mm, N, MPa, t/mm³")
    assumed = st.get("understand", {}).get("data", {}).get("assumptions")
    pg.row("assumptions applied", "\n".join(f"• {x}" for x in assumed) if assumed
           else "none: the parameters were given in full")

    # -- 2 results
    pg.heading("2   Results")
    if post.get("max_disp") is not None:
        pg.row("max. displacement", f"|U| = {post['max_disp']:.4g} mm")
    if post.get("max_vm") is not None:
        pg.row("max. von Mises stress", f"{post['max_vm']:.4g} MPa, {_peak_note(is_plate)}")

    # -- 3 verification
    pg.heading("3   Verification against a reference solution")
    case = get_case(v["kb_ref"]) if v.get("kb_ref") else None
    if case is None:
        pg.row("reference", "no applicable reference in the knowledge base: verdict N/A, "
               "nothing fabricated")
    else:
        unit = "" if str(v.get("quantity", "")).startswith("Kt") else " mm"
        num = lambda x: "—" if x is None else f"{x:.5g}{unit}"
        pg.row("reference", f"{case.name} [{case.id}]")
        pg.row("source", case.source)
        pg.row("valid for", case.scope_note)
        pg.row("compared quantity", str(v.get("quantity", "")))
        pg.row("reference value", num(v.get("ref")))
        pg.row("FE value", num(v.get("fem")))
        pg.row("relative error", f"{v.get('error_pct')} %   (tolerance ≤ {v.get('tol_pct')} %)")
        pg.row("verdict", verdict, color=STATUS_C.get(verdict, INK))
        if rec.get("diagnosis"):
            pg.note("The dashboard's diagnosis of this failure is not part of this report: "
                    "a language model wrote it.")

    # -- 4 validity
    pg.heading("4   Validity of the linear-elastic model")
    pg.note("Checked beside the verdict, not part of it: the solver and the reference share these "
            "assumptions, so they can agree with each other where both are wrong.")
    for c in checks:
        status = {"PASS": "OK", "WARNING": "EXCEEDED"}.get(c["status"], "NOT CHECKED")
        pg.row(c["name"], f"{status}: {c['message']}", color=STATUS_C[status])

    # -- 5 computation time
    step = BODY * 1.45 / 72
    pg.heading("5   Computation time", keep=(len(rec["stages"]) + 1) * step)
    pg.put(LEFT, "stage", color=MUTED)
    pg.put(LEFT + 2.9, "wall time", color=MUTED, ha="right")
    pg.put(LEFT + 3.15, "status", color=MUTED)
    pg.y += step
    solve = st.get("solve", {}).get("data", {})
    for s in rec["stages"]:
        secs = s.get("seconds")
        label = s.get("label", s["id"])
        if s["id"] == "solve" and solve.get("threads"):
            label += f" (CalculiX, {solve['threads']} threads)"
        pg.put(LEFT, label)
        pg.put(LEFT + 2.9, "—" if secs is None else f"{secs:.2f} s", ha="right")
        pg.put(LEFT + 3.15, s.get("status", ""), color=STATUS_C.get(s.get("status", "").upper(), INK))
        pg.y += step
    pg.y += 0.06
    chain = [st[k].get("seconds") for k in CHAIN if k in st]
    if chain and None not in chain:
        pg.row("FE analysis", f"{sum(chain):.2f} s (pre-process to verify)")
    if rec.get("seconds"):
        pg.row("end to end", f"{rec['seconds']:.2f} s (all stages)")
    pg.row("machine", meta.get("host", "—"))
    if any(s.get("seconds") is None for s in rec["stages"]):
        pg.note("—: not timed in this record. A recorded example keeps its recorded conversation; "
                "only its FE analysis was re-run and timed.")

    # -- 6 figures
    figures = [(Path(a["path"]), f"mesh, {meta.get('n_elem', pre.get('elements'))} elements")
               for a in st.get("preprocess", {}).get("artifacts", []) if a.get("kind") == "image"]
    figures += [(Path(path), _VIEWS.get(key, key))
                for a in st.get("postprocess", {}).get("artifacts", []) if a.get("kind") == "views"
                for key, path in a["views"].items()]
    figures = [(f if f.is_absolute() else HERE / f, c) for f, c in figures]
    figures = [(f, c) for f, c in figures if f.exists()]
    if figures:
        pg.new_page()
        pg.heading("6   Figures")
        gap = 0.25
        cw = (TEXT_W - gap) / 2
        for k in range(0, len(figures), 2):
            pair = [(imread(f), c) for f, c in figures[k:k + 2]]
            h = max(cw * img.shape[0] / img.shape[1] for img, _ in pair)
            pg.need(h + 0.35)
            for j, (img, cap) in enumerate(pair):
                x = LEFT + j * (cw + gap)
                ih = cw * img.shape[0] / img.shape[1]
                ax = pg.fig.add_axes([x / W, 1 - (pg.y + ih) / H, cw / W, ih / H])
                ax.imshow(img, interpolation="none")
                ax.set_axis_off()
                pg.put(x, f"Fig. {k + j + 1}   {cap}", size=SMALL, dy=h + 0.06, color=MUTED)
            pg.y += h + 0.4

    # -- footer on every page
    for n, fig in enumerate(pg.figs, 1):
        y = (BOTTOM - 0.35) / H
        fig.add_artist(Line2D([LEFT / W, 1 - LEFT / W], [y + 0.18 / H] * 2, color=RULE, lw=0.6))
        fig.text(LEFT / W, y, f"AgenticFEA · {ident(rec)} · written by deterministic code from the "
                 "run record, not by a language model",
                 fontsize=SMALL - 0.6, color=MUTED, va="top")
        fig.text(1 - LEFT / W, y, f"{n} / {len(pg.figs)}", fontsize=SMALL - 0.6, color=MUTED,
                 va="top", ha="right")

    buf = io.BytesIO()
    with matplotlib.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        # No creation date: the same record gives the same file.
        with PdfPages(buf, metadata={"Title": f"Technical report: {title}", "Author": "AgenticFEA",
                                     "CreationDate": None}) as pdf:
            for fig in pg.figs:
                pdf.savefig(fig)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_json", help="a run record: out/runs/<id>/live_run.json or demo/<name>/run.json")
    ap.add_argument("-o", "--out", help="output PDF (default: FEA-report-<run>.pdf here)")
    args = ap.parse_args()
    rec = json.loads(Path(args.run_json).read_text())
    out = Path(args.out or file_name(rec))
    out.write_bytes(build(rec))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
