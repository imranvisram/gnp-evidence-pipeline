# GNP evidence pipeline

Ingests the five GNP executive interview files and produces a themed evidence matrix,
an automated verbatim-verification report, and a grounded Q&A that declines questions
the interviews cannot answer.

**Run it:** `pip install -r requirements.txt && export ANTHROPIC_API_KEY=... && streamlit run app.py`

CLI equivalent: `python pipeline.py run` writes `output/evidence_matrix.md` and
`output/verification_report.md`; `python pipeline.py ask "your question"` queries it.

## Design

Three stages, one of which uses a model.

| Stage | What it does | Model? |
| --- | --- | --- |
| Extract | One call per interview file returns candidate quotes tagged to a fixed theme list, with a flag for reported speech vs interviewer shorthand | Yes |
| Verify | Exact substring match of every quote against its source file after normalizing curly quotes, dash variants and whitespace. Wording and case are not normalized. Failures are reported, not hidden | No |
| Ask | Answers from verified quotes only, then re-verifies every cited quote before display. No surviving quote means the answer is "Not in the interviews." | Yes, bounded |

Three choices worth naming:

1. **Verification is deterministic.** A model checking its own quotes is not
   verification. Stage 2 is plain Python string matching and nothing else.
2. **Themes are set by hand, not discovered.** Letting the model theme each file
   independently produces five incompatible taxonomies. The tool finds and proves
   evidence for themes a human decided mattered.
3. **Speech is distinguished from notes.** These files are largely the interviewer's
   shorthand. Only spans inside quotation marks are reported speech, and a deck that
   attributes a paraphrase to a named executive is not client-ready.

## Limits

- Extraction recall is not guaranteed. The tool proves that what it surfaced is real;
  it does not prove nothing was missed.
- Themes are fixed at seven. A genuinely novel theme in a new file would land in the
  closest existing bucket.
- Verification confirms a quote exists in the file. It cannot confirm the interviewer
  recorded the speaker accurately.
