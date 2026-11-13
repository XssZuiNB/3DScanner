---
name: grilling
description: Grill the user relentlessly about a plan, decision, or idea. Use when the user wants to stress-test their thinking, or uses any 'grill' trigger phrases.
---

Interview the user relentlessly until you reach a shared understanding. Map this as a **design tree**: every decision branches into the decisions that hang off it.

Work the tree in **rounds**. The **frontier** is every decision whose prerequisites are already settled: the questions you can ask _now_ without guessing at answers you haven't heard yet. Ask the whole frontier in one round through the AskUserQuestion dialog, then wait for the user's answers before the next round.

Shape each question for the dialog:

- **Self-contained**: the dialog can hide the text before it, so the question itself names the decision and carries its evidence in a line or two of plain words.
- **The options are the choices**: two to four, each description saying what that choice keeps, deletes or costs, your recommendation first and marked recommended. The dialog adds the free-text answer itself.
- **Four questions a dialog**: a larger frontier goes into consecutive dialogs within the same round.

Where the dialog is unavailable (a subagent, a non-interactive session), ask the round as numbered text questions, each with your recommended answer.

Each round the user answers reshapes the tree: settled decisions push the frontier outward and unblock questions that depended on them. Recompute the frontier and ask the next round. A question whose answer depends on another question still open in this round belongs to a _later_ round, not this one.

Finding _facts_ is your job, never the user's. When a frontier question needs a fact from the environment (filesystem, tools, etc.), dispatch a sub-agent on sonnet to find it (a lookup); don't ask the user for anything you could look up yourself. Don't block on it: a running exploration is an unsettled prerequisite, so only the questions downstream of it wait for the sub-agent to report; ask the rest of the frontier now. The _decisions_ are the user's: put each to them and wait.

The session is done when the frontier is empty: every branch of the design tree visited, nothing left silently assumed. Do not act on it until the user confirms you have reached a shared understanding.
