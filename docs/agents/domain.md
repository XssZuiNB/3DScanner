# Domain docs

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

How the workflow skills in `.claude/skills/` read this repo's domain documentation.

## Before exploring, read these

- `GLOSSARY.md`: the project's terms, one context, grouped by area.
- `docs/adr/`: the ADRs that touch the area you are about to work in.

Where either does not exist yet, proceed silently: the `domain-modeling` skill creates them
when the first term or decision is resolved.

## Use the glossary's vocabulary

When your output names a domain concept (a ticket title, a refactor proposal, a hypothesis,
a test name, an identifier), use the term as `GLOSSARY.md` defines it, never a synonym it
lists under _Avoid_. A concept the glossary lacks is either language the project does not
use (reconsider) or a gap (note it for `domain-modeling`).

## Flag ADR conflicts

If your output contradicts an ADR, say so explicitly rather than silently overriding it:

> _Contradicts ADR-0003 (the camera is hand-held; there is no turntable), but worth reopening because…_
