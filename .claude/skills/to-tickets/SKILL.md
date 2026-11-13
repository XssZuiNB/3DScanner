---
name: to-tickets
description: Break a plan, spec, or the current conversation into a set of tracer-bullet tickets, each declaring its blocking edges, one file per ticket in the local tracker.
disable-model-invocation: true
---

# To Tickets

Break a plan, spec, or conversation into a set of **tickets**: tracer-bullet vertical slices, each declaring the tickets that **block** it.

The issue tracker and its status vocabulary are described in `docs/agents/issue-tracker.md`.

## Process

### 1. Gather context

Work from whatever is already in the conversation context. If the user passes a reference (a spec path, an issue number or URL) as an argument, fetch it and read its full body and comments.

### 2. Explore the codebase (optional)

If you have not already explored the codebase, do so to understand the current state of the code. Ticket titles and descriptions should use the project's domain glossary vocabulary, and respect ADRs in the area you're touching.

Look for opportunities to prefactor the code to make the implementation easier. "Make the change easy, then make the easy change."

### 3. Draft vertical slices

Break the work into **tracer bullet** tickets.

<vertical-slice-rules>

- Each slice cuts a narrow but COMPLETE path through every layer (schema, API, UI, tests): vertical, NOT a horizontal slice of one layer
- A completed slice is demoable or verifiable on its own
- Each slice is sized to about one hour of an agent's work, and fits in a single fresh context window; merge a slice that small and that close to another into it
- Each slice says how it is proven offline: the seam, the simulation or dry run that exercises it, and which frozen files may change and how
- Any prefactoring should be done first

</vertical-slice-rules>

Give each ticket its **blocking edges**: the other tickets that must complete before it can start. A ticket with no blockers can start immediately.

Give each ticket its **model**, by who catches its mistakes. `opus` by default: whenever a mistake can pass `tools/verify.sh` and show only on the hardware, or cost a reset there (what AGENTS.md §3 lists, threads and locks, a format another program reads), and every slice whose risk is in doubt. `sonnet` only where opus is overkill: mechanical work whose every mistake the offline proof catches (a move, a deletion, a document, a tool, behaviour kept and shown by the frozen traces and the dry-run comparison). The model runs the ticket's plan, its implementation and its review.

**Wide refactors are the exception to vertical slicing.** A **wide refactor** is one mechanical change (rename a column, retype a shared symbol) whose **blast radius** fans across the whole codebase, so a single edit breaks thousands of call sites at once and no vertical slice can land green. Don't force it into a tracer bullet; sequence it as **expand–contract**. First expand: add the new form beside the old so nothing breaks. Then migrate the call sites over in batches sized by blast radius (per package, per directory), each batch its own ticket blocked by the expand, keeping CI green batch to batch because the old form still exists. Finally contract: delete the old form once no caller remains, in a ticket blocked by every migrate batch. When even the batches can't stay green alone, keep the sequence but let them share an integration branch that all block a final integrate-and-verify ticket; green is promised only there.

### 4. Quiz the user

Present the proposed breakdown as a numbered list. For each ticket, show:

- **Slug**: the ticket's name, a short kebab-case summary of its task (step 5)
- **Title**: short descriptive name
- **Blocked by**: which other tickets (if any) must complete first
- **Model**: opus or sonnet, and what decides it
- **What it delivers**: the end-to-end behaviour this ticket makes work

Ask the user:

- Does the granularity feel right? (too coarse / too fine)
- Are the blocking edges correct: does each ticket only depend on tickets that genuinely gate it?
- Should any tickets be merged or split further?

Iterate until the user approves the breakdown.

### 5. Publish the tickets to the configured tracker

Publish the approved tickets to the local tracker `docs/agents/issue-tracker.md` describes: one file per ticket under `.scratch/<feature-slug>/issues/<ticket-slug>.md`, named by its slug: a short kebab-case summary of its task, unique across the project and never reused (that document says how to check). Each file's "Blocked by" lists the slugs it depends on. Use the per-ticket file template below: one ticket per file, never a single combined file.

Work the **frontier**: any ticket whose blockers are all done. For a purely linear chain that means top to bottom.

Then write the stage brief, `.scratch/<feature-slug>/brief.md`, from [../implement/BRIEF-TEMPLATE.md](../implement/BRIEF-TEMPLATE.md): every implementer and reviewer of these tickets reads it first.

Do NOT close or modify any parent issue.

<local-ticket-template>

# <ticket-slug>: <Ticket title>

**What to build:** the end-to-end behaviour this ticket makes work, from the user's perspective, not a layer-by-layer implementation list.

**Blocked by:** the slugs of the tickets that gate this one, or "None (can start immediately)".

**Status:** ready-for-agent

**Model:** opus or sonnet, and what decides it (step 3).

**Test seam:** how the behaviour is proven offline (the seam, the dry run or simulation means that reaches it), and which frozen files may change and how; "none" when no frozen file may change.

- [ ] Acceptance criterion 1
- [ ] Acceptance criterion 2

</local-ticket-template>

Avoid specific file paths or code snippets: they go stale fast. Exception: if a prototype produced a snippet that encodes a decision more precisely than prose can (state machine, reducer, schema, type shape), inline it and note briefly that it came from a prototype. Trim to the decision-rich parts, not a working demo, just the important bits.
