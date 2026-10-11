#!/usr/bin/env python3
"""Agentic FEA -- control-room dashboard (Streamlit).

Reads a uniform run record (pipeline_run.run_pipeline live, or a demo/ snapshot
in DEMO_MODE) and renders a clickable walk through the eight layers: each stage
shows its agent-style action flow (left) and its artifacts (right), with the
verbatim acceptance report one click away.

Modes:
  live      (default)  -- type a problem, run the real pipeline (needs fea + key);
                          a switch in the header also replays the recorded examples
  DEMO_MODE=1          -- replay only: the pre-recorded demo/ snapshots (no key, no solver)

Shared use: live runs can be limited (at once, per user per day) and old run
directories pruned through environment settings -- see run_guard.py.

Design: a control room, not a SaaS. Dark, monospace, three signal colours --
amber = running, green = passed, red = verification failed.

Run:  streamlit run app.py          (live)
      DEMO_MODE=1 streamlit run app.py   (demo)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

import paraview_open
import run_guard
import viewer
from paths import RunPaths

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE / "demo"
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"     # replay only, no live switch
DEFAULT_DESC = ("A steel cantilever beam 100 mm long, 10x10 mm square section, "
                "fixed at one end, with a 1000 N downward load at the free end.")

st.set_page_config(page_title="Agentic FEA", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
:root{ --amber:#e8a13a; --green:#3fb950; --red:#f85149; --warn:#d29922;
  --muted:#6e7681; --line:#232a35; --panel:#11161f; --bg:#0a0e14; --text:#c9d1d9; }
html,body,[class*="css"],.stMarkdown{ font-family:ui-monospace,"SF Mono",Menlo,monospace; }
.block-container{ padding-top:1.6rem; max-width:1500px; }
#MainMenu,footer,header{ visibility:hidden; }
.muted{color:var(--muted)} .amber{color:var(--amber)} .green{color:var(--green)} .red{color:var(--red)}
.brand{ letter-spacing:.35em; color:var(--muted); font-size:.78rem; }
.quote{ color:var(--text); font-size:.8rem; line-height:1.5; text-align:right; }
.query{ color:var(--text); font-size:.8rem; line-height:1.5; margin-top:.3rem; }
.brow{ display:flex; margin:.6rem 0; }
.brow.user{ justify-content:flex-end; }
.brow.agent{ justify-content:flex-start; }
.bub{ display:inline-block; max-width:82%; padding:.42rem .75rem; border-radius:11px; font-size:.8rem; line-height:1.45; white-space:pre-wrap; }
.bub.user{ background:#1b2533; border:1px solid #2a3a4f; }
.bub.agent{ background:transparent; border:1px solid var(--line); }
.bub.ok{ background:transparent; border:1px solid var(--green); }
div[data-testid="stFormSubmitButton"]{ display:none; }
hr.rule{ border:none; border-top:1px solid var(--line); margin:.5rem 0 1.1rem; }
.stagebar{ display:flex; justify-content:space-between; margin:.2rem 0 .4rem; }
.stage{ display:flex; flex-direction:column; align-items:center; flex:1; position:relative; }
.stage .conn{ position:absolute; top:20px; left:50%; width:100%; height:2px; background:var(--line); z-index:0; }
.stage:last-child .conn{ display:none; }
.node{ width:40px; height:40px; border-radius:50%; border:2px solid var(--line); background:var(--bg);
  display:flex; align-items:center; justify-content:center; font-size:.95rem; color:var(--muted); z-index:1; }
.node.pass{ border-color:var(--green); color:var(--green); }
.node.fail{ border-color:var(--red); color:var(--red); }
.node.run{ border-color:var(--amber); color:var(--amber); box-shadow:0 0 0 4px rgba(232,161,58,.12); }
/* selected = keep the status colour (green/red); show "viewing" with a neutral
   focus halo only -- colour encodes status, the halo encodes selection. */
.node.sel{ box-shadow:0 0 0 3px rgba(201,209,217,.16), 0 0 12px 3px rgba(201,209,217,.10); }
.slabel{ margin-top:.45rem; font-size:.68rem; color:var(--text); }
.sdetail{ font-size:.62rem; color:var(--muted); min-height:.8rem; }
.panel-h{ display:flex; justify-content:space-between; color:var(--muted); font-size:.7rem;
  letter-spacing:.14em; margin:.1rem 0 .5rem; }
.tag{ border:1px solid var(--line); border-radius:3px; padding:.03rem .4rem; font-size:.64rem; }
.tag.green{border-color:var(--green);color:var(--green)} .tag.red{border-color:var(--red);color:var(--red)}
.tag.amber{border-color:var(--amber);color:var(--amber)}
.sub{ display:flex; justify-content:space-between; gap:.6rem; padding:.32rem 0; border-bottom:1px solid var(--line); }
.sub .t{ color:var(--text); font-size:.78rem; } .sub .d{ color:var(--muted); font-size:.7rem; }
.sub .s{ font-size:.72rem; min-width:1rem; text-align:right; }
.s.pass{color:var(--green)} .s.fail{color:var(--red)} .s.warning{color:var(--warn)}
.kv{ display:flex; justify-content:space-between; padding:.28rem 0; border-bottom:1px solid var(--line); font-size:.8rem; }
.kv .k{color:var(--muted)} .kv .v{color:var(--text)}
.note{ color:var(--muted); font-size:.69rem; line-height:1.5; margin-top:.5rem; }
.bigverdict{ font-size:1.1rem; letter-spacing:.1em; }
.statusbar{ color:var(--muted); font-size:.7rem; border-top:1px solid var(--line); padding-top:.7rem; margin-top:1rem; }
.diag{ border:1px solid var(--red); border-radius:6px; padding:.7rem .9rem; }
.stButton button{ width:100%; font-family:ui-monospace,monospace; font-size:.68rem; color:var(--text);
  background:transparent; border:1px solid var(--line); }
.stButton button:hover{ border-color:var(--amber); color:var(--amber); }
/* report expanders -- quiet & secondary: muted grey, small, with breathing room
   above. They are open-on-demand, so they should not pull the eye at rest. */
[data-testid="stExpander"]{ margin-top:.95rem; }
[data-testid="stExpander"] details, [data-testid="stExpander"] summary{ border-color:var(--line); }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *{ font-size:.7rem !important; color:var(--muted) !important; }
[data-testid="stExpander"] summary svg{ fill:var(--muted); width:.82rem; height:.82rem; }
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:hover *{ color:var(--text) !important; }
/* clickable stage bar: each node is a Streamlit button captured into a ring.
   border + glyph inherit currentColor (set per-node), so one colour paints both;
   opaque bg masks the connector line that runs behind the row of circles. */
div[class*="st-key-stagebtn"] button{ width:40px; height:40px; min-height:40px; padding:0;
  margin:.1rem auto 0; border-radius:50%; border:2px solid currentColor; background:var(--bg);
  font-size:.95rem; line-height:1; display:flex; align-items:center; justify-content:center; }
/* keep the disc opaque on hover too, so the connector never bleeds through */
div[class*="st-key-stagebtn"] button:hover{ background:var(--bg); }
/* Centre each circle in its column. Streamlit left-packs the shrink-wrapped
   button container, so the real lever is the column's vertical stack: align-items
   centre pushes the disc to the middle, under its (already centred) title. */
.snode-cap{ text-align:center; }
.st-key-stagebar_row [data-testid="stVerticalBlock"]{ gap:.25rem; overflow:visible; }
.st-key-stagebar_row [data-testid="stColumn"] [data-testid="stVerticalBlock"]{ align-items:center; }
.st-key-stagebar_row [data-testid="stColumn"]{ position:relative; z-index:1; overflow:visible; }
.st-key-stagebar_row div[class*="st-key-stagebtn"]{ width:fit-content; overflow:visible; }
.st-key-stagebar_row .stButton{ width:auto; }
.st-key-stagebar_row [data-testid="stHorizontalBlock"]{ position:relative; }
.st-key-stagebar_row [data-testid="stHorizontalBlock"]::before{ content:""; position:absolute;
  top:21px; left:7.3%; right:7.3%; height:2px; background:var(--line); z-index:0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

_SYM = {"pass": "✓", "fail": "!", "running": "●", "queued": ""}


def _img(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else HERE / p)


@st.cache_data(show_spinner=False)
def _viewer_figure(path: str, mtime: float):
    """Built once per file; `mtime` makes a re-exported viewer.npz a new entry."""
    return viewer.figure(viewer.load_payload(path))


def load_demo(name: str) -> dict:
    return json.loads((DEMO_DIR / name / "run.json").read_text())


def load_demo_chat(name: str) -> list:
    """Pre-recorded layer-1 clarification thread (record_chat.py) as (role, text) turns."""
    p = DEMO_DIR / name / "chat.json"
    return [(t["role"], t["text"]) for t in json.loads(p.read_text())] if p.exists() else []


def demo_examples() -> list:
    if not DEMO_DIR.exists():
        return []
    return sorted(p.name for p in DEMO_DIR.iterdir() if (p / "run.json").exists())


def stage_bar_html(stage_list: list, sel=None) -> str:
    cells = []
    for i, s in enumerate(stage_list):
        klass = {"pass": "pass", "fail": "fail", "running": "run"}.get(s["status"], "")
        if sel is not None and i == sel:
            klass += " sel"
        sym = _SYM.get(s["status"], str(i + 1))
        cells.append(f'<div class="stage"><div class="conn"></div>'
                     f'<div class="node {klass}">{sym}</div>'
                     f'<div class="slabel">{s["label"]}</div>'
                     f'<div class="sdetail">{s.get("detail","")}</div></div>')
    return f'<div class="stagebar">{"".join(cells)}</div>'


# ---------------- command console: a clarification chat (live) ----------------
for _k, _v in {"run": None, "stage": 0, "chat": [], "since": 0,
               "phase": "chatting", "completed": None, "thinking": False,
               "demo_chat": [], "demo_step": 0,
               "mode": "replay" if DEMO_MODE else "live"}.items():
    st.session_state.setdefault(_k, _v)
REPLAY = st.session_state.mode == "replay"


def _switch_mode():
    """Each mode starts clean: a fresh chat, or the first recorded example."""
    for k, v in {"run": None, "stage": 0, "chat": [], "since": 0, "phase": "chatting",
                 "completed": None, "thinking": False, "pick": None,
                 "demo_chat": [], "demo_step": 0}.items():
        st.session_state[k] = v


def _branch():
    """Layer 1 on the conversation so far (IN-THREAD, LLM only). The user's
    message is already in the chat; this reads the conversation since the
    current problem began and appends the reply.

    The chat is handed over as a conversation, not glued into one string, so
    the model can see which question each answer replies to -- and the reply's
    meta records what it asked the user to settle (task_understanding.resolve).
    """
    import task_understanding as tu
    chat = st.session_state.chat
    try:
        step = tu.understand(chat[st.session_state.since:])
    except Exception as exc:  # noqa: BLE001
        chat.append(("agent", run_guard.BUDGET_MESSAGE if run_guard.budget_exhausted(exc)
                     else f"(could not reach the model: {exc.__class__.__name__})"))
        return
    chat.append(("agent", step.message, {"asks": step.asks}))
    if step.kind in ("off_topic", "out_of_scope"):
        st.session_state.since = len(chat)        # the next message starts a new problem
    elif step.kind == "ready":
        st.session_state.completed = step.completed
        st.session_state.phase = "ready"


def _bubble(role: str, text: str):
    import html
    cls = "user" if role == "user" else ("ok" if text.startswith("✓") else "agent")
    st.markdown(f'<div class="brow {role}"><div class="bub {cls}">{html.escape(text)}</div></div>',
                unsafe_allow_html=True)


with st.container(border=True):
    hc1, hc2 = st.columns([3, 1])
    hc1.markdown('<div class="brand">AGENTIC&middot;FEA</div>', unsafe_allow_html=True)
    if DEMO_MODE:
        hc2.markdown('<div class="muted" style="text-align:right;font-size:.72rem;margin-top:.15rem">'
                     'mode: DEMO · replay</div>', unsafe_allow_html=True)
    else:
        hc2.radio("mode", ["live", "replay"], key="mode", horizontal=True, on_change=_switch_mode,
                  label_visibility="collapsed",
                  format_func={"live": "live · chat", "replay": "replay · recorded"}.get)
    if REPLAY:
        examples = demo_examples()
        if not examples:
            st.error("no demo snapshots found — run: python record_demo.py")
            st.stop()
        pick = st.selectbox("example", examples, label_visibility="collapsed")
        if st.session_state.get("pick") != pick:
            st.session_state.run = load_demo(pick)
            st.session_state.pick = pick
            st.session_state.stage = 0
            st.session_state.demo_chat = load_demo_chat(pick)
            st.session_state.demo_step = 1 if st.session_state.demo_chat else 0
        dchat, dstep = st.session_state.demo_chat, st.session_state.demo_step
        for _role, _text in dchat[:dstep]:
            _bubble(_role, _text)
        if dstep < len(dchat):                       # still stepping through the recorded clarification
            if st.button("▶ next"):
                st.session_state.demo_step = dstep + 1
                st.rerun()
        elif st.session_state.run:                   # thread done -> show the resolved query
            st.markdown(f'<div class="query">&ldquo;{st.session_state.run["description"]}&rdquo;</div>',
                        unsafe_allow_html=True)
    else:
        for _role, _text, *_ in st.session_state.chat:
            _bubble(_role, _text)
        if st.session_state.thinking:
            with st.spinner("understanding…"):           # shown ABOVE the input, like agent replies
                _branch()
            st.session_state.thinking = False
            st.rerun()
        elif st.session_state.phase == "chatting":
            with st.form("chat_form", clear_on_submit=True, border=False):
                reply = st.text_input("reply", label_visibility="collapsed",
                                      placeholder="describe your beam problem, or answer the question…")
                if st.form_submit_button("send") and reply.strip():
                    st.session_state.chat.append(("user", reply.strip()))
                    st.session_state.thinking = True
                    st.rerun()
        elif st.session_state.phase == "done":
            if st.button("↻ new problem"):
                st.session_state.run = None
                st.session_state.chat = []
                st.session_state.since = 0
                st.session_state.completed = None
                st.session_state.phase = "chatting"
                st.session_state.stage = 0
                st.rerun()

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ---------------- live run: stream the pipeline once understanding is complete ----------------
if not REPLAY and st.session_state.phase == "ready" and st.session_state.run is None:
    import subprocess
    import sys
    defs = [("understand", "Understand"), ("parameterize", "Parameterize"),
            ("preprocess", "Pre-Process"), ("solve", "Solve"), ("physics", "Physics"),
            ("postprocess", "Post-Process"), ("verify", "Verify")]
    status = {sid: "queued" for sid, _ in defs}
    bar, cap = st.empty(), st.empty()

    def _draw():
        bar.markdown(stage_bar_html([{"label": lbl, "status": status[sid], "detail": ""}
                                     for sid, lbl in defs]), unsafe_allow_html=True)

    _draw()

    def _refuse(message: str):
        st.session_state.phase = "chatting"
        st.session_state.chat.append(("agent", message))
        st.rerun()

    # Shared-use limits (run_guard; all off unless configured).
    user = run_guard.user_of(st.context.headers)
    run_guard.prune_old_runs()
    if run_guard.runs_left(user) == 0:
        _refuse("You have used today's live runs. The recorded examples still work: "
                "switch to replay above.")
    with st.spinner("waiting for a free run slot…"), run_guard.run_slot() as got:
        if not got:
            _refuse("Every run slot is busy right now; please send the problem again in a minute.")
        run_guard.count_run(user)
        # Every live run gets its own directory. Two sessions at once -- two
        # browser tabs on the same dashboard -- would otherwise overwrite each
        # other's mesh, results and run record without any error.
        run_paths = RunPaths.new_run()
        live_json = run_paths.live_run_json
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "pipeline_run.py"), "--stream",
             "--workdir", str(run_paths.root), "--out", str(live_json),
             st.session_state.completed],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(HERE))
        buf = []
        for line in proc.stdout:
            buf.append(line)
            if line.startswith("@@ "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] in status:
                    status[parts[1]] = parts[2]
                    cap.markdown(f'<div class="muted" style="font-size:.72rem">▶ {parts[1]} … {parts[2]}</div>',
                                 unsafe_allow_html=True)
                    _draw()
        proc.wait()
    if live_json.exists() and proc.returncode == 0:
        st.session_state.run = json.loads(live_json.read_text())
        st.session_state.phase = "done"
        st.session_state.stage = 0
        st.rerun()
    elif run_guard.budget_exhausted("".join(buf)):
        _refuse(run_guard.BUDGET_MESSAGE)
    else:
        st.session_state.phase = "chatting"
        st.session_state.chat.append(("agent", "the pipeline failed — see the terminal; try adjusting the problem."))
        st.error("pipeline failed:\n\n" + "".join(buf[-25:]))
        st.stop()

if not REPLAY and st.session_state.phase == "chatting":
    st.stop()   # still clarifying — nothing below the divider yet
if REPLAY and st.session_state.demo_step < len(st.session_state.demo_chat):
    st.stop()   # still stepping through the recorded clarification — dashboard not yet

run = st.session_state.run
if run is None:
    st.info("Describe a beam problem above to begin.")
    st.stop()

stages = run["stages"]
st.session_state.stage = min(st.session_state.stage, len(stages) - 1)

# ---------------- stage bar (each circle IS a clickable button) ----------------
sel = st.session_state.stage

# Streamlit >=1.39 tags every widget container with a .st-key-<key> class, so a
# button can be styled into the node itself -- no more detached selector row. The
# per-node colour/halo is dynamic (status + selection), so we emit it here. Border
# and glyph both read from currentColor, so one colour rule paints the whole ring.
_NODE_C = {"pass": "var(--green)", "fail": "var(--red)", "running": "var(--amber)"}
_css = []
for i, s in enumerate(stages):
    c = _NODE_C.get(s["status"], "var(--muted)")
    _css.append(f".st-key-stagebtn{i} button{{ color:{c}; }}")
    hover = f"color:{c}; border-color:{c};"
    if i != sel:                       # faint hover ring; the selected node keeps its strong halo
        hover += " box-shadow:0 0 0 3px rgba(201,209,217,.12);"
    _css.append(f".st-key-stagebtn{i} button:hover{{ {hover} }}")
# selected = a layered neutral halo (bright inner ring + faint outer ring + soft bloom),
# prominent enough to read clearly as "you are viewing this one"
_css.append(f".st-key-stagebtn{sel} button{{ box-shadow:0 0 0 4px rgba(201,209,217,.34),"
            f"0 0 0 7px rgba(201,209,217,.15),0 0 20px 7px rgba(201,209,217,.20); }}")
st.markdown("<style>" + "".join(_css) + "</style>", unsafe_allow_html=True)

with st.container(key="stagebar_row"):
    cols = st.columns(len(stages))
    for i, s in enumerate(stages):
        with cols[i]:
            if st.button(_SYM.get(s["status"], str(i + 1)), key=f"stagebtn{i}"):
                st.session_state.stage = i
                st.rerun()
            st.markdown(f'<div class="snode-cap"><div class="slabel">{s["label"]}</div>'
                        f'<div class="sdetail">{s.get("detail","")}</div></div>',
                        unsafe_allow_html=True)

stage = stages[sel]

# ---------------- body: left actions / right artifacts ----------------
left, right = st.columns([1, 1.15], gap="large")

with left:
    with st.container(border=True):
        tag = {"pass": "green", "fail": "red"}.get(stage["status"], "amber")
        st.markdown(f'<div class="panel-h"><span>{sel+1:02d} &nbsp;{stage["label"].upper()}</span>'
                    f'<span class="tag {tag}">{stage["status"].upper()}</span></div>', unsafe_allow_html=True)
        for a in stage.get("actions", []):
            s = a.get("status", "pass")
            st.markdown(f'<div class="sub"><div><div class="t">{a["text"]}</div>'
                        f'<div class="d">{a.get("data","")}</div></div>'
                        f'<div class="s {s}">{_SYM.get(s,"")}</div></div>', unsafe_allow_html=True)


def render_artifact(a: dict):
    kind = a.get("kind")
    if a.get("title"):
        st.markdown(f'<div class="panel-h"><span>~ &nbsp;{a["title"].upper()}</span></div>',
                    unsafe_allow_html=True)
    if kind == "image":
        p = Path(_img(a["path"]))
        if p.exists():
            st.image(_img(a["path"]), width="stretch")
        else:
            st.caption("(no image)")
    elif kind == "views":
        # The options come from the artifact itself, not from a fixed list: the
        # two geometries render different things (a 3D solid is shown from three
        # camera directions, a flat plate as its stress components in one top
        # view), so hard-coding one geometry's keys makes the other one crash.
        views = a.get("views", {})
        pretty = {"z": "σ · Z (top)", "y": "σ · Y (side)", "x": "σ · X (section)",
                  "disp": "displacement"}
        keys = list(views)
        labels = [pretty.get(k, k) for k in keys]
        if not keys:
            st.caption("(no views)")
            return
        # Keying on the option set keeps a previous geometry's selection from
        # leaking in when the user switches demos.
        choice = st.radio("view", labels, horizontal=True, label_visibility="collapsed",
                          key="views_" + "_".join(keys))
        src = views[keys[labels.index(choice)]]
        if Path(_img(src)).exists():
            st.image(_img(src), width="stretch")
        else:
            st.caption("(no image)")
    elif kind == "viewer3d":
        # Real element edges over the result field: drag to rotate, scroll to
        # zoom, hover a node for its value. Field and deformed shape are menus
        # inside the figure (see viewer.figure for why not widgets here). The
        # PNG renders below stay as the fixed views.
        p = Path(_img(a["path"]))
        if not p.exists():
            st.caption("(no 3D view for this run)")
            return
        st.plotly_chart(_viewer_figure(str(p), p.stat().st_mtime), theme=None, width="stretch",
                        config=viewer.PLOTLY_CONFIG, key=f"vfig_{p}")
        # Local only: offered when the browser is on this machine, ParaView is
        # installed here and the run kept its .vtu (see paraview_open).
        exe = paraview_open.available(p.parent, st.context.ip_address)
        if exe and st.button(f"open in {paraview_open.paraview_name(exe)}", key=f"pv_{p}",
                             help="Starts ParaView on this machine with this run's .vtu, "
                                  "set up like the view above."):
            try:
                paraview_open.launch(RunPaths(p.parent), exe)
                st.toast(f"{paraview_open.paraview_name(exe)} is starting with this run")
            except OSError as exc:
                st.error(f"could not start ParaView: {exc}")
    elif kind == "code":
        st.code(a.get("text", ""), language="text")
    elif kind == "panel":
        st.markdown(f'<div class="note" style="font-size:.8rem;color:var(--text)">{a.get("text","")}</div>',
                    unsafe_allow_html=True)
    elif kind == "list":
        for item in a.get("items", []):
            st.markdown(f'<div class="sub"><div class="t">• {item}</div></div>', unsafe_allow_html=True)
    elif kind == "checks":
        for c in a.get("items", []):
            s = c["status"].lower()
            star = " *" if c.get("critical") else ""
            st.markdown(f'<div class="sub"><div><div class="t">{c["name"]}{star}</div>'
                        f'<div class="d">{c["message"]}</div></div>'
                        f'<div class="s {s}">{_SYM.get(s, c["status"][0])}</div></div>', unsafe_allow_html=True)
    elif kind == "judge":
        v = a["verify"]
        for k, val in [("δ analytical (PL³/3EI)", f'{v.get("delta_ref")} mm'),
                       ("δ solver (FEM)", f'{v.get("delta_fem")} mm'),
                       ("relative error", f'{v.get("error_pct")} %'),
                       ("threshold", f'≤ {v.get("tol_pct")} %')]:
            st.markdown(f'<div class="kv"><span class="k">{k}</span><span class="v">{val}</span></div>',
                        unsafe_allow_html=True)
        st.markdown(
            f'<div class="note">KB ref <span class="amber">{v.get("kb_ref")}</span> &middot; {v.get("provenance","")}'
            f'<br>reference <b>selected</b> by RAG + deterministic gate; error <b>computed by deterministic '
            f'code</b> — the LLM never invents the formula.</div>', unsafe_allow_html=True)


with right:
    with st.container(border=True):
        arts = stage.get("artifacts", [])
        for a in arts:
            render_artifact(a)
        if stage.get("report"):
            with st.expander("▸ full acceptance report"):
                st.code(stage["report"], language="text")
        if not arts and not stage.get("report"):
            st.caption("no artifact for this stage")
    # diagnostics on the failing stage
    if stage["status"] == "fail" and run.get("diagnosis"):
        d = run["diagnosis"]
        with st.container(border=True):
            st.markdown('<div class="panel-h"><span>⚠ &nbsp;HUMAN REVIEW REPORT (diagnostics)</span>'
                        f'<span class="tag red">{d.get("failure_class","")}</span></div>', unsafe_allow_html=True)
            for k, val in [("root cause", d.get("root_cause", "")),
                           ("confidence", d.get("confidence", "")),
                           ("suggested fix", d.get("suggested_fix", "")),
                           ("next action", d.get("recommended_next_action", ""))]:
                st.markdown(f'<div class="note"><span class="amber">{k}:</span> {val}</div>',
                            unsafe_allow_html=True)
            st.markdown('<div class="note">advice only — nothing is executed automatically (human decides).</div>',
                        unsafe_allow_html=True)
            if d.get("report"):
                with st.expander("▸ full diagnostic report"):
                    st.code(d["report"], language="text")

# ---------------- status bar ----------------
m = run.get("meta", {})
verdict = run.get("verdict", "—")
vcol = {"PASS": "green", "FAIL": "red"}.get(verdict, "amber")
st.markdown(
    f'<div class="statusbar">verdict: <span class="{vcol}">{verdict}</span> &nbsp;&nbsp; '
    f'solver: {m.get("solver","—")} &nbsp;&nbsp; units: {m.get("units","SI-mm")} &nbsp;&nbsp; '
    f'mesh: {m.get("n_elem","—")} elem / {m.get("n_nodes","—")} nodes &nbsp;&nbsp; '
    f'host: {m.get("host","—")}</div>', unsafe_allow_html=True)
