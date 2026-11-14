---
name: implement
description: "Implement a piece of work based on a spec or set of tickets."
disable-model-invocation: true
---

Implement the work described by the user in the spec or tickets.

If the user passes a ticket reference, fetch it from the issue tracker and state its title before starting. If the reference is ambiguous, ask.

Start only a ticket whose `Status:` is `ready-for-agent` and whose blockers are all `done`; otherwise tell the user which status or blocker stands in the way and stop. A `ready-for-human` ticket belongs to the user: prepare its offline part (the procedure, the exact commands, what to watch and when to stop) and hand it over.

**Read the stage brief first** (`.scratch/<feature>/brief.md`; [BRIEF-TEMPLATE.md](BRIEF-TEMPLATE.md) says what it holds), then the ticket with its Comments. Open another document only where the brief or the ticket points to it.

**Plan check.** Before writing code, report a plan of at most 15 lines: the seam and how it is tested offline, the frozen files the change may alter and how, and every decision the ticket leaves to you with the choice you would make. When an orchestrator runs you, wait for its answer; otherwise go on.

Call the Skill tool with "tdd" where possible, at pre-agreed seams.

Build with `tools/build.sh` regularly (the warning set is part of the standards), run the check at the seam you are working on regularly, and `tools/verify.sh` once at the end (AGENTS.md §4).

**Verify by the change's risk**, with the repo's verification tools rather than scripts of your own:

- behaviour kept: the frozen traces, and the dry-run comparison against the base commit;
- a new rule, refusal or branch: its test cases, and one to three mutants that show each one bites;
- documents only: the greps the ticket names, and the command examples run as written.

The hardware is live and shared. Work offline: build, test, dry runs.

**Stop and ask.** Only the user gives a go for the hardware, puts hands on the hardware, types a password or settles a decision. Stop, ask the user (with the AskUserQuestion tool where it is available) and wait when:

- a step touches the hardware (AGENTS.md §3) and no standing authorisation written in AGENTS.md covers it;
- the ticket, the spec or an ADR leaves a decision open, or the work would contradict one of them;
- a command needs root (AGENTS.md §2);
- a stop condition from AGENTS.md §3 appears.

Ask once, with what is needed, why, and the exact command or the choices. Record the question and the answer under the ticket's `## Comments`. A hardware step the user has not yet allowed goes on the list of live checks the work still needs.

**Finish** on an uncommitted working tree. Append one section of at most 30 lines to the ticket's `## Comments`: what was built, the decisions taken, how it was verified (the commands and their results), the live checks it still needs. Report in at most 15 lines; the details are in the ticket and the diff.

**The ticket's model.** An orchestrator runs the implementer and its reviewer on the model the ticket's `Model:` line names (to-tickets), with the effort, both passed explicitly: on opus the implementer at effort xhigh and the reviewer at high, on sonnet both at high. A ticket without the line runs on opus. A sonnet implementer whose work reaches what that line rules out (AGENTS.md §3's hardware, a thread, a lock, a format another program reads) stops, says so, and the ticket goes to opus; so does a sonnet ticket that fails `tools/verify.sh` twice or whose review finds a correctness defect.

**Review by the change's risk**, always by someone who did not write it; you fix what the review finds:

- a small change with a mechanical proof (under about 100 changed lines besides regenerated frozen files, adding no rule, refusal, branch, thread, lock or wire field): the orchestrator's check;
- a deletion proven by the build and by a grep for every deleted name, whatever its line count, adding no rule, refusal, branch, thread, lock or wire field: the orchestrator's check;
- documents only: the orchestrator's check.
- everything else: one fresh reviewer (`two-axis-review`) on the ticket's model, handed the results already run and the change's risk points, so it builds nothing and reads only what settles a point. Run alone, call the Skill tool with "two-axis-review" once done; it reviews in a fresh context.

The orchestrator's check reads the whole diff, confirms the ticket's proof (the frozen-file diff exactly as the test seam names it, the comparison, the mutant, the greps) and asks the passed-tree check of `tools/verify.sh` (AGENTS.md §4) whether the working tree as it is has passed, instead of running verify again. A change after that pass needs a new pass. Run alone, the user's look before the commit is that check.

When the tier is in doubt, the fresh reviewer. The user commits, and the ticket's `Status:` becomes `done` when the user accepts the change. In a run the user asked to continue on its own, the orchestrator sets `done` and commits each checked ticket locally once the passed-tree check says yes, one commit per ticket whose subject starts with `Ticket <ticket-slug>:`, never pushing; small related tickets may go to one implementer, each still its own commit.
