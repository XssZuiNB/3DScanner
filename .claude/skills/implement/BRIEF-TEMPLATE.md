# Stage brief: template

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

A stage brief is what every implementer and reviewer of a stage reads first, instead of
the whole document set. The orchestrator writes it at the stage's start (after
`/to-tickets`), as `.scratch/<feature>/brief.md`, and adds one line to "Done so far" at each
ticket's commit. It stays under 150 lines: it points to documents, it does not copy them.

## The sections

1. **Hard rules** (at most 10 lines): what may not be run, signalled or written on the
   live hardware; the offline commands that are allowed; where temporary files go; that git
   state is the orchestrator's. Copied from AGENTS.md §1-§3 in their shortest true form,
   with the section named.
2. **The stage's goal** (at most 5 lines) and the spec and ADRs it implements, by path.
3. **Where things are** (a table, at most 20 rows): the module or file for each concern the
   stage touches, as AGENTS.md §7 rule 5's homes table gives it.
4. **Interfaces that must not move silently** (at most 15 lines): the log lines tools read,
   the dump keys, the options, the frozen files, and the test that pins each.
5. **Standards in one screen** (at most 15 lines): the comment rule, the style per kind of
   file, the evidence tags, the glossary's avoided words that come up in this stage.
6. **Verification** (at most 10 lines): the exact commands of the repo's verification
   tools (build, the verify target, the dry-run comparison against a base commit, the mutant
   runner), and the proof each kind of change needs (implement's "Verify by the change's
   risk").
7. **Done so far** (one line per committed ticket): its slug, its commit, and what it
   changed that a later ticket must know.
8. **Known gaps and open decisions** (one line each, with the ticket that holds it).
