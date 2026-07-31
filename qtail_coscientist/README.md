# Q-TAIL AI Co-Scientist — Multi-Agent Paper Revision Framework

Adapted from the open-source **AI co-scientist** architecture
(Google AI co-scientist → LLNL `open-ai-co-scientist`, MIT, LLNL-CODE-2010270).

We instantiate the six specialized agents plus a Supervisor to **co-author the revision of the
Q-TAIL paper** (ver.24 → ver.25) using the latest project data from the running Q-TAIL-MVP service.
This mirrors the scientific-method reasoning loop: generate → reflect → rank → evolve → proximity →
meta-review, with test-time-compute scaling via Elo-ranked tournaments.

## Agent roster (roles)
| Agent | Role in this revision |
|-------|------------------------|
| **Supervisor** | Parses the revision goal, queues work, allocates sources, integrates outputs (main session). |
| **Generation** | Drafts the full enhanced manuscript (LaTeX) from ver.24 + new Open X real-data evidence. |
| **Reflection** | Reviews for scientific accuracy, internal consistency, fabrication risk, missing sections. |
| **Ranking** | Elo tournament comparing candidate drafts (ver.24 baseline vs Generation vs Evolution). |
| **Evolution** | Combines top draft + critiques into an improved (evolved) version. |
| **Proximity** | Measures alignment between the draft and the source JSON/project data (number drift check). |
| **Meta-review** | Holistic final QA: novelty, impact, clarity, claim-boundary honesty, venue readiness. |

## How it was executed
The open-source framework normally drives a Gradio UI over an OpenRouter LLM. In this environment the
orchestration was executed with LLM **sub-agents** that each embodied one agent role and read the shared
`paper_v25/source_brief.md` + `experiment_numbers`. The Supervisor (main agent) ran the Elo-ranked
evolution cycle and compiled the final PDF + Word deliverables. `agents.py` documents the same agent
prompts and supervisor loop as a reusable blueprint.

## Run artifacts
- `agents.py` — agent prompt templates + supervisor loop blueprint.
- `../paper_v25/source_brief.md` — consolidated source (ver.24 + new data).
- `../paper_v25/experiment_numbers.md` — exact numbers table.
- `../paper_v25/src/paper_v25.tex` — generated manuscript.
- `../paper_v25/build/` — compiled PDF + DOCX.
