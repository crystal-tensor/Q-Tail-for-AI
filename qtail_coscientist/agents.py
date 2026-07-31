"""
Q-TAIL AI Co-Scientist — agent roster & supervisor blueprint
=============================================================
Adapted from the open-source AI co-scientist (LLNL/open-ai-co-scientist, Google AI co-scientist).
The six specialized agents + a Supervisor implement a scientific-method reasoning loop used here to
co-author the Q-TAIL paper revision (ver.24 -> ver.25).

In this environment the orchestration was executed with LLM sub-agents, each embodying one role and
reading `../paper_v25/source_brief.md`. The Supervisor (main agent) ran the Elo-ranked evolution cycle
and compiled the PDF + DOCX. This module documents the same prompts/loop as a reusable blueprint and is
importable (`from agents import AGENTS, SUPERVISOR_PROMPT`).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str
    tool_use: bool = False

# ---- Shared context contract for every agent ----
SHARED = (
    "Read ../paper_v25/source_brief.md (ver.24 structure + NEW Open X real-data evidence) and "
    "../paper_v25/experiment_numbers.md (exact figures). Honor all claim-boundary rules: the Open X "
    "results are allocation-head training convergence + same-budget DATA-DISTRIBUTION quality gains, "
    "NOT end-to-end robot-policy success. Never invent metrics; use the brief's exact numbers."
)

AGENTS = {
    "generation": Agent(
        "Generation", "Draft the enhanced manuscript",
        f"{SHARED} You are the GENERATION agent. Produce a complete, compilable LaTeX manuscript "
        "(article class, English, ~12-16 pages) that KEEPS the ver.24 theoretical framework (PT prior, "
        "three mappings, theorems) and MetaWorld simulation results, and ADDS a new 'Real Open X "
        "Embodiment Validation' section using the new data (171.62 GiB / 562 shards / 2071 episodes; "
        "tail allocation 6.07x; tail success +5.41pp rel +11.31%; CVaR +5.56pp; paired bootstrap p<1e-3) "
        "plus the production data-service section and a multi-agent methodology note. Use figures "
        "fig1/fig7/fig8/fig9/fig10/fig11/real_hardware_pt/fig_pt_distribution_validation. Output the .tex.",
        tool_use=True),

    "reflection": Agent(
        "Reflection", "Critique for correctness & gaps",
        f"{SHARED} You are the REFLECTION agent. Review the draft for: (1) scientific accuracy of the PT/"
        "optimal-transport math; (2) every number matching the brief; (3) claim-boundary honesty; "
        "(4) missing sections; (5) LaTeX errors. Return a prioritized critique (file), not a rewrite.",
        tool_use=True),

    "proximity": Agent(
        "Proximity", "Measure draft<->source alignment",
        f"{SHARED} You are the PROXIMITY agent. Compare the draft against the source JSON/project data. "
        "Flag any number drift, misattribution (Quafu/Baihua vs Open X), or inconsistency. Output an "
        "alignment report (file) with pass/fail per metric.",
        tool_use=True),

    "ranking": Agent(
        "Ranking", "Elo-rank candidate drafts",
        f"You are the RANKING agent. Given candidate drafts (ver.24 baseline, Generation draft, Evolution "
        "draft), run an Elo tournament on quality/novelty/rigor and report the ranked order with scores.",
        tool_use=False),

    "evolution": Agent(
        "Evolution", "Improve top draft using critiques",
        f"{SHARED} You are the EVOLUTION agent. Take the top-ranked draft and the Reflection + Proximity "
        "critiques; produce an improved (evolved) manuscript that resolves every actionable critique while "
        "preserving the framework. Output the revised .tex.",
        tool_use=True),

    "meta_review": Agent(
        "Meta-Review", "Final holistic QA",
        f"{SHARED} You are the META-REVIEW agent. Holistic final check: novelty, impact, clarity, "
        "claim-boundary honesty, venue readiness, and that the multi-agent process is documented. "
        "Approve or list must-fix items (file).",
        tool_use=True),
}

SUPERVISOR_PROMPT = (
    "You are the SUPERVISOR of the Q-TAIL AI co-scientist. Parse the revision goal, assign the six "
    "specialist agents to the worker queue, allocate the shared source brief, run the Elo-ranked "
    "evolution cycle (Generation -> Reflection+Proximity -> Evolution -> Meta-review), and integrate the "
    "winning draft into the final PDF + DOCX. Scale test-time compute by iterating the cycle until the "
    "Meta-review passes."
)

def run_cycle(generation_fn: Callable, reflection_fn: Callable, evolution_fn: Callable,
              meta_review_fn: Callable, max_iters: int = 3) -> str:
    """Blueprint supervisor loop. In this run the functions were realized by LLM sub-agents."""
    draft = generation_fn()
    for _ in range(max_iters):
        critique = reflection_fn(draft)
        evolved = evolution_fn(draft, critique)
        verdict = meta_review_fn(evolved)
        if verdict.passed:
            return evolved
        draft = evolved
    return draft
