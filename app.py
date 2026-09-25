"""
Streamlit front end for the GNP evidence pipeline.

Run:  streamlit run app.py

The last run is cached to output/quotes.json, so reopening the app restores the
previous result without spending another extraction pass.
"""

import json
import os

import streamlit as st

import pipeline as pl

UPLOAD_CACHE = pl.OUTPUT_DIR / "uploads"

st.set_page_config(
    page_title="GNP evidence pipeline",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- styling -----------------------------------------------------------------
# One accent colour, generous line height, quotes set apart from UI chrome.
st.markdown(
    """
    <style>
      :root {
        --accent: #C9663A;
        --edge: rgba(255,255,255,0.10);
        --muted: rgba(255,255,255,0.55);
      }
      .block-container { padding-top: 2.5rem; max-width: 1180px; }
      h1 { font-weight: 600 !important; letter-spacing: -0.02em; margin-bottom: .2rem !important; }
      .lede { color: var(--muted); font-size: .95rem; margin-bottom: 2rem; }

      .statrow { display: flex; gap: .75rem; margin-bottom: 2rem; flex-wrap: wrap; }
      .stat { flex: 1; min-width: 150px; border: 1px solid var(--edge);
              border-radius: 10px; padding: 1rem 1.1rem; }
      .stat .n { font-size: 2rem; font-weight: 600; line-height: 1.1; }
      .stat .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .09em;
                 color: var(--muted); margin-top: .35rem; }
      .stat.accent { border-color: var(--accent); }
      .stat.accent .n { color: var(--accent); }

      .quote { border-left: 2px solid var(--accent); padding: .1rem 0 .1rem 1rem;
               margin: 1rem 0; }
      .quote .t { font-size: 1.02rem; line-height: 1.6; }
      .quote .a { font-size: .8rem; color: var(--muted); margin-top: .45rem; }
      .chip { display: inline-block; font-size: .68rem; letter-spacing: .05em;
              text-transform: uppercase; border: 1px solid var(--edge);
              border-radius: 20px; padding: .1rem .55rem; margin-left: .4rem; }
      .stTabs [data-baseweb="tab"] { font-size: .95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- key ---------------------------------------------------------------------
try:
    key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    key = os.environ.get("ANTHROPIC_API_KEY", "")

with st.sidebar:
    st.markdown("### Setup")
    if not key:
        key = st.text_input("Anthropic API key", type="password")
    uploaded = st.file_uploader(
        "Add interview files",
        type="txt",
        accept_multiple_files=True,
        help="Added to the five bundled interviews. A file with the same name replaces it.",
    )
    run = st.button("Run pipeline", type="primary", use_container_width=True)
    st.caption("Results are cached to disk - reopening restores your last run.")
    cached = sorted(UPLOAD_CACHE.glob("*.txt")) if UPLOAD_CACHE.exists() else []
    if cached:
        st.markdown("**Added files**")
        for path in cached:
            st.caption(f"- {path.name}")
        if st.button("Remove added files", use_container_width=True):
            removed = {path.name for path in cached}
            for path in cached:
                path.unlink()
            if pl.QUOTES_PATH.exists():
                qs = json.loads(pl.QUOTES_PATH.read_text(encoding="utf-8"))
                qs = [q for q in qs if q.get("source_file") not in removed]
                pl.QUOTES_PATH.write_text(json.dumps(qs, indent=2), encoding="utf-8")
            st.session_state.clear()
            st.rerun()

if key:
    os.environ["ANTHROPIC_API_KEY"] = key


def all_sources():
    """Bundled interviews plus any previously uploaded file still cached on disk."""
    sources = pl.load_sources()
    if UPLOAD_CACHE.exists():
        for path in sorted(UPLOAD_CACHE.glob("*.txt")):
            sources[path.name] = path.read_text(encoding="utf-8")
    return sources


def get_sources():
    """Corpus for this run: bundled + cached uploads + anything just uploaded.

    Newly uploaded files are written to disk so verification still has a source
    to check against after a refresh. Same filename means replace.
    """
    sources = all_sources()
    for f in uploaded or []:
        text = f.getvalue().decode("utf-8")
        UPLOAD_CACHE.mkdir(parents=True, exist_ok=True)
        (UPLOAD_CACHE / f.name).write_text(text, encoding="utf-8")
        sources[f.name] = text
    return sources


def restore():
    """Reload the last run from disk so a refresh costs nothing."""
    if not pl.QUOTES_PATH.exists():
        return False
    try:
        quotes = json.loads(pl.QUOTES_PATH.read_text(encoding="utf-8"))
        sources = all_sources()
        verified, failed = pl.verify_all(quotes, sources)
        st.session_state.update(
            sources=sources, verified=verified, failed=failed, total=len(quotes)
        )
        return True
    except Exception:
        return False


st.title("GNP Foundation - evidence pipeline")
st.markdown(
    '<div class="lede">Ingests the executive interview notes, extracts themed '
    "evidence, and proves every quote against its source file before it is shown."
    "</div>",
    unsafe_allow_html=True,
)

# --- run ---------------------------------------------------------------------
if run:
    if not key:
        st.error("Add an API key in the sidebar first.")
        st.stop()
    sources = get_sources()
    client = pl.get_client()
    quotes = []
    bar = st.progress(0.0, text="Extracting ...")
    for i, (name, content) in enumerate(sources.items(), start=1):
        bar.progress(i / len(sources), text=f"Extracting {name}")
        try:
            quotes.extend(pl.extract_from_file(client, name, content))
        except Exception as exc:
            st.warning(f"{name} failed: {exc}")
    bar.empty()
    verified, failed = pl.verify_all(quotes, sources)
    pl.OUTPUT_DIR.mkdir(exist_ok=True)
    pl.QUOTES_PATH.write_text(json.dumps(quotes, indent=2), encoding="utf-8")
    st.session_state.update(
        sources=sources, verified=verified, failed=failed, total=len(quotes)
    )

if "verified" not in st.session_state and not restore():
    st.info("Add your key and press **Run pipeline** to process the interview files.")
    st.stop()

sources = st.session_state["sources"]
verified = st.session_state["verified"]
failed = st.session_state["failed"]
total = st.session_state["total"]
rate = f"{round(100 * len(verified) / total)}%" if total else "-"

st.markdown(
    f"""<div class="statrow">
      <div class="stat"><div class="n">{len(sources)}</div><div class="k">Interviews ingested</div></div>
      <div class="stat"><div class="n">{total}</div><div class="k">Quotes extracted</div></div>
      <div class="stat accent"><div class="n">{len(verified)}</div><div class="k">Verified word-for-word</div></div>
      <div class="stat"><div class="n">{len(failed)}</div><div class="k">Failed and excluded</div></div>
      <div class="stat"><div class="n">{rate}</div><div class="k">Verification rate</div></div>
    </div>""",
    unsafe_allow_html=True,
)

tab_ask, tab_matrix, tab_verify = st.tabs(
    ["Ask the interviews", "Evidence matrix", "Verification report"]
)


def quote_card(q):
    kind = "speech" if q["direct_speech"] else "note"
    st.markdown(
        f'<div class="quote"><div class="t">"{q["quote"]}"</div>'
        f'<div class="a">{q["speaker"]}'
        f'<span class="chip">{kind}</span>'
        f'<span class="chip">{q["source_file"]}</span></div></div>',
        unsafe_allow_html=True,
    )


# --- ask ---------------------------------------------------------------------
with tab_ask:
    st.caption(
        "Answers draw only on quotes that passed verification, and every quote is "
        "re-checked against its source file before it appears here."
    )
    question = st.text_input(
        "Question",
        placeholder="What do grantees say about the granting process?",
        label_visibility="collapsed",
    )
    if st.button("Ask", type="primary") and question:
        with st.spinner("Searching the verified corpus ..."):
            result = pl.cmd_ask(question, verified, sources)
        if result["answer"] == pl.REFUSAL:
            st.warning(f"**{pl.REFUSAL}**")
            st.caption(result.get("reason", ""))
        else:
            st.markdown(f"#### {result['answer']}")
            if result.get("shortlisted"):
                st.caption(
                    f"{result['shortlisted']} quotes screened as relevant - "
                    f"{len(result['quotes'])} cited - all re-verified"
                )
            for q in result["quotes"]:
                quote_card(q)

# --- matrix ------------------------------------------------------------------
with tab_matrix:
    present = [
        (k, v) for k, v in pl.THEMES.items() if any(q["theme"] == k for q in verified)
    ]
    labels = [
        f"{v.split(' - ')[0]} ({sum(q['theme'] == k for q in verified)})"
        for k, v in present
    ]
    choice = st.radio("Theme", labels, horizontal=True, label_visibility="collapsed")
    theme_key = present[labels.index(choice)][0]
    st.caption(pl.THEMES[theme_key])
    for q in [q for q in verified if q["theme"] == theme_key]:
        quote_card(q)
    st.divider()
    st.download_button(
        "Download full matrix (markdown)",
        pl.write_matrix(verified),
        file_name="evidence_matrix.md",
    )

# --- verification ------------------------------------------------------------
with tab_verify:
    report = pl.write_verification_report(verified, failed)
    st.markdown(report)
    st.download_button(
        "Download verification report", report, file_name="verification_report.md"
    )
