"""
GNP evidence pipeline.

Three stages, one of which uses a model:
  1. extract  - LLM pulls candidate quotes out of each interview file (one call per file)
  2. verify   - pure Python exact-match check of every quote against its source file
  3. ask      - answers questions from VERIFIED quotes only, re-verifies before display

Stage 2 never calls a model. A model checking its own quotes is not verification.

CLI:
  python pipeline.py extract              # interviews/ -> output/quotes.json
  python pipeline.py verify               # quotes.json -> output/verification_report.md
  python pipeline.py matrix               # quotes.json -> output/evidence_matrix.md
  python pipeline.py ask "your question"
  python pipeline.py run                  # extract + verify + matrix
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
INTERVIEW_DIR = ROOT / "interviews"
OUTPUT_DIR = ROOT / "output"
QUOTES_PATH = OUTPUT_DIR / "quotes.json"

MODEL = "claude-sonnet-5"
MAX_TOKENS = 4000

# Themes are set by the consultant, not discovered by the model. Letting the model
# invent themes per file produces five incompatible sets. The tool's job is to find
# and prove the evidence for themes a human decided mattered.
THEMES = {
    "decision_rights": "Decisions stall - escalation to the top, circular review, no stated ownership of decisions",
    "grantee_experience": "Grantee experience - slow grants, hard to work with, fragmented face to the community",
    "silos_handoffs": "Silos and handoffs - teams not working across boundaries, late involvement, protected knowledge",
    "change_readiness": "Change readiness and resistance - scars, scepticism, resistors, 'flavour of the year'",
    "communication": "Internal communication - rumour mill, unannounced changes, no supporting plan",
    "capability_gaps": "Capability and skill gaps - generalists, workplanning, coaching, communication skills",
    "prior_change": "Evidence GNP has changed successfully before - what worked, and the mechanism that made it work",
    "strengths": "Strengths to preserve - people, values, mission, community engagement model",
}

def speaker_for(filename: str) -> str:
    """Known files get their exact title; anything new is derived from the filename."""
    if filename in SPEAKERS:
        return SPEAKERS[filename]
    stem = filename.rsplit(".", 1)[0]
    parts = [w for w in stem.split("_") if w.lower() != "interview" and not w.isdigit()]
    return " ".join(parts).replace(" and ", " & ") if parts else filename


SPEAKERS = {
    "interview_1_President_and_CEO.txt": "President & CEO",
    "interview_2_Chief_Operating_Officer.txt": "Chief Operating Officer",
    "interview_3_Head_of_Learning.txt": "Head of Learning",
    "interview_4_Head_of_Org_Effectiveness.txt": "Head of Org Effectiveness",
    "interview_5_Project_Manager.txt": "Project Manager",
}


# ---------------------------------------------------------------- stage 2: verify

def normalize(text: str) -> str:
    """Fold away the differences that are noise, keep the ones that are meaning.

    Curly quotes, en/em dashes and non-breaking spaces vary between how a model
    echoes a span and how it sits in the file. Wording and case do not get folded:
    if the words differ, the quote fails.
    """
    text = unicodedata.normalize("NFKC", text)
    for ch in "\u2018\u2019\u02bc\u2032":
        text = text.replace(ch, "'")
    for ch in "\u201c\u201d\u2033":
        text = text.replace(ch, '"')
    for ch in "\u2013\u2014\u2212":
        text = text.replace(ch, "-")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def verify_quote(quote: str, source_text: str) -> dict:
    """Exact substring match after normalization. No model, no fuzzy matching."""
    n_quote = normalize(quote)
    n_source = normalize(source_text)

    if not n_quote:
        return {"status": "fail", "reason": "empty quote", "offset": None}

    offset = n_source.find(n_quote)
    if offset != -1:
        return {"status": "verified", "reason": "exact match", "offset": offset}

    # Classify the failure so the report says something useful.
    if n_source.lower().find(n_quote.lower()) != -1:
        reason = "matches only if case is ignored"
    elif "..." in n_quote or "\u2026" in quote:
        reason = "contains an ellipsis - not a contiguous span"
    else:
        reason = "no match in source - paraphrase or fabrication"
    return {"status": "fail", "reason": reason, "offset": None}


def load_sources() -> dict:
    sources = {}
    for path in sorted(INTERVIEW_DIR.glob("*.txt")):
        sources[path.name] = path.read_text(encoding="utf-8")
    if not sources:
        sys.exit(f"No .txt files found in {INTERVIEW_DIR}")
    return sources


def verify_all(quotes: list, sources: dict) -> tuple:
    """Returns (verified_quotes, failed_quotes)."""
    verified, failed = [], []
    for q in quotes:
        source_name = q.get("source_file")
        if source_name not in sources:
            q["verification"] = {
                "status": "fail",
                "reason": f"source file '{source_name}' not found",
                "offset": None,
            }
            failed.append(q)
            continue
        q["verification"] = verify_quote(q["quote"], sources[source_name])
        (verified if q["verification"]["status"] == "verified" else failed).append(q)
    return verified, failed


def write_verification_report(verified: list, failed: list) -> str:
    total = len(verified) + len(failed)
    lines = [
        "# Verbatim verification report",
        "",
        f"**{total} quotes extracted - {len(verified)} verified word-for-word "
        f"- {len(failed)} failed.**",
        "",
        "Every quote is checked by exact substring match against its source file "
        "after normalizing curly quotes, dash variants and whitespace. Wording and "
        "case are not normalized. No model is involved in this stage.",
        "",
    ]
    if failed:
        lines += ["## Failures (excluded from the evidence matrix)", ""]
        for q in failed:
            lines += [
                f"- **{q.get('source_file', 'unknown')}** - {q['verification']['reason']}",
                f"  > {q['quote']}",
                "",
            ]
    else:
        lines += ["No failures. Every extracted quote appears verbatim in its source file.", ""]

    lines += ["## Verified by source file", ""]
    by_file = {}
    for q in verified:
        by_file.setdefault(q["source_file"], []).append(q)
    for name in sorted(by_file):
        lines.append(f"- {name}: {len(by_file[name])} verified")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------- stage 1: extract

EXTRACT_PROMPT = """You are extracting evidence from one interview note file for a \
consulting case-for-change.

Return every line or span that is strong evidence for one of these themes:

{theme_block}

Rules, and the tool downstream checks all of them mechanically:
- Copy each span EXACTLY as it appears in the file, character for character. Do not \
fix typos, do not tidy grammar, do not join separated lines, do not use ellipses.
- Each span must be contiguous text from the file and at most 40 words.
- Do not include the leading "- " bullet marker.
- Set direct_speech to true only when the span sits inside double quotation marks in \
the file (actual reported speech). These notes are mostly the interviewer's own \
shorthand, which is not the same thing as a quotation, and the deck needs to know \
which is which.
- Pick the strongest evidence, not everything. Aim for 8 to 14 spans.

Return ONLY a JSON array, no prose, no code fence. Each element:
{{"theme": "<one theme key>", "quote": "<exact span>", "direct_speech": true|false, \
"note": "<max 12 words on why this is evidence>"}}

The file is {filename}. Its contents follow.

<file>
{content}
</file>"""


def extract_from_file(client, filename: str, content: str) -> list:
    theme_block = "\n".join(f"- {k}: {v}" for k, v in THEMES.items())
    prompt = EXTRACT_PROMPT.format(
        theme_block=theme_block, filename=filename, content=content
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  ! {filename}: model did not return valid JSON ({exc})")
        return []

    # The model occasionally returns a nested list instead of a flat array.
    # Flatten one level and ignore anything that is not a dict rather than
    # crashing the whole run on one malformed file.
    flat = []
    for item in items if isinstance(items, list) else [items]:
        flat.extend(item) if isinstance(item, list) else flat.append(item)

    out = []
    for item in flat:
        if not isinstance(item, dict) or item.get("theme") not in THEMES:
            continue
        out.append(
            {
                "theme": item["theme"],
                "quote": item.get("quote", ""),
                "direct_speech": bool(item.get("direct_speech")),
                "note": item.get("note", ""),
                "source_file": filename,
                "speaker": speaker_for(filename),
            }
        )
    return out


def get_client():
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment.")
    return Anthropic()


def cmd_extract() -> list:
    client = get_client()
    sources = load_sources()
    all_quotes = []
    for name, content in sources.items():
        print(f"  extracting {name} ...")
        try:
            quotes = extract_from_file(client, name, content)
        except Exception as exc:
            print(f"    ! failed: {exc} - continuing with the other files")
            quotes = []
        print(f"    {len(quotes)} candidate quotes")
        all_quotes.extend(quotes)
    OUTPUT_DIR.mkdir(exist_ok=True)
    QUOTES_PATH.write_text(json.dumps(all_quotes, indent=2), encoding="utf-8")
    print(f"  wrote {len(all_quotes)} candidates to {QUOTES_PATH}")
    return all_quotes


# ------------------------------------------------------------------ evidence matrix

def write_matrix(verified: list) -> str:
    lines = [
        "# Themed evidence matrix",
        "",
        "Verified quotes only. Every quote below passed an exact match against its "
        "source file. `speech` marks reported speech inside quotation marks; the rest "
        "is the interviewer's own note text.",
        "",
    ]
    for key, description in THEMES.items():
        items = [q for q in verified if q["theme"] == key]
        if not items:
            continue
        lines += [f"## {description}", "", f"*{len(items)} verified quotes*", ""]
        lines += ["| Speaker | Evidence | Type | Source |", "| --- | --- | --- | --- |"]
        for q in items:
            quote = q["quote"].replace("|", "\\|")
            kind = "speech" if q["direct_speech"] else "note"
            lines.append(f"| {q['speaker']} | {quote} | {kind} | {q['source_file']} |")
        lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------- stage 3: ask
#
# Two calls, one job each. A single call had to judge relevance AND enforce the
# no-fabrication rule at the same time, and the strictness needed for the second
# job suppressed the first: abstractly-phrased questions got declined even when
# the evidence was sitting in the corpus. Splitting them makes a refusal mean
# something specific - nothing in the corpus bears on the question - rather than
# "the model did not feel confident".

RELEVANCE_PROMPT = """Below are numbered quotes from executive interviews at a \
foundation. Your ONLY job is to list which quotes bear on the question.

Be generous. Include a quote if it is partial evidence, background, a concrete \
example of the thing asked about, or a counterpoint to it. Match on meaning, not \
wording: the question may use abstract or general terms where the quotes are \
specific and concrete, or the reverse.

You are NOT deciding whether the question can be answered. Do not filter for \
sufficiency. Another step does that.

Return ONLY JSON: {{"ids": [<numbers>]}}   (an empty list if genuinely nothing relates)

<quotes>
{corpus}
</quotes>

Question: {question}"""

ANSWER_PROMPT = """You answer a question using ONLY the quotes below. They have \
already been screened as potentially relevant, so judge whether they actually \
support an answer.

If they do: answer in at most four sentences and cite the id of EVERY quote that \
supports what you said. Where two quotes make the same point differently, or one \
names a mechanism another only gestures at, cite both.
If they do not: set "answer" to null.

Do not reason from general knowledge. Do not introduce any fact, cause or \
conclusion that no quote states. Do not soften a gap into a partial answer. You \
MAY combine quotes, including comparing speakers who conflict - synthesis is \
expected, invention is not.

Return ONLY JSON: {{"answer": "<text>" or null, "quote_ids": [<ids>]}}

<quotes>
{corpus}
</quotes>

Question: {question}"""

REFUSAL = "Not in the interviews."


def _json_from(response) -> dict:
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def cmd_ask(question: str, verified: list, sources: dict) -> dict:
    client = get_client()

    # Pass 1 - relevance only, cannot refuse.
    full_corpus = "\n".join(
        f'[{i}] {q["speaker"]}: "{q["quote"]}"' for i, q in enumerate(verified)
    )
    shortlist_raw = _json_from(
        client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": RELEVANCE_PROMPT.format(
                        corpus=full_corpus, question=question
                    ),
                }
            ],
        )
    ).get("ids", [])

    shortlist = [
        i for i in shortlist_raw if isinstance(i, int) and 0 <= i < len(verified)
    ]
    if not shortlist:
        return {
            "answer": REFUSAL,
            "quotes": [],
            "reason": "no quote in the corpus bears on this question",
            "shortlisted": 0,
        }

    # Pass 2 - answer strictly from the shortlist.
    short_corpus = "\n".join(
        f'[{i}] {verified[i]["speaker"]}: "{verified[i]["quote"]}"' for i in shortlist
    )
    result = _json_from(
        client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(
                        corpus=short_corpus, question=question
                    ),
                }
            ],
        )
    )

    if not result.get("answer"):
        return {
            "answer": REFUSAL,
            "quotes": [],
            "reason": f"{len(shortlist)} quotes were related but none supported an answer",
            "shortlisted": len(shortlist),
        }

    # Re-verify every cited quote against its source file before display.
    cited, dropped = [], []
    for i in result.get("quote_ids", []):
        if not isinstance(i, int) or not 0 <= i < len(verified):
            dropped.append(str(i))
            continue
        q = verified[i]
        if verify_quote(q["quote"], sources[q["source_file"]])["status"] == "verified":
            cited.append(q)
        else:
            dropped.append(str(i))

    if not cited:
        return {
            "answer": REFUSAL,
            "quotes": [],
            "reason": "no cited quote survived re-verification",
            "shortlisted": len(shortlist),
        }
    return {
        "answer": result["answer"],
        "quotes": cited,
        "dropped": dropped,
        "shortlisted": len(shortlist),
    }


# ----------------------------------------------------------------------------- cli

def load_quotes() -> list:
    if not QUOTES_PATH.exists():
        sys.exit(f"{QUOTES_PATH} not found - run `python pipeline.py extract` first.")
    return json.loads(QUOTES_PATH.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="GNP evidence pipeline")
    parser.add_argument(
        "command", choices=["extract", "verify", "matrix", "ask", "run"]
    )
    parser.add_argument("question", nargs="?", help="for the ask command")
    parser.add_argument("--quotes", help="path to a quotes json (for verify)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    sources = load_sources()

    if args.command in ("extract", "run"):
        quotes = cmd_extract()
    elif args.quotes:
        quotes = json.loads(Path(args.quotes).read_text(encoding="utf-8"))
    else:
        quotes = load_quotes()

    if args.command == "ask":
        verified, _ = verify_all(quotes, sources)
        result = cmd_ask(args.question or "", verified, sources)
        print("\n" + result["answer"] + "\n")
        for q in result["quotes"]:
            print(f'  {q["speaker"]}: "{q["quote"]}"  [{q["source_file"]}]')
        return

    verified, failed = verify_all(quotes, sources)
    report = write_verification_report(verified, failed)
    (OUTPUT_DIR / "verification_report.md").write_text(report, encoding="utf-8")
    print("\n" + report.split("\n\n")[1])

    if args.command in ("matrix", "run"):
        (OUTPUT_DIR / "evidence_matrix.md").write_text(
            write_matrix(verified), encoding="utf-8"
        )
        print(f"Wrote {OUTPUT_DIR / 'evidence_matrix.md'}")


if __name__ == "__main__":
    main()
