# Issue tracker: local markdown

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

Specs and tickets live as markdown files in `.scratch/` at the repository root. `.gitignore`
keeps them out of git: they are working files. `ROADMAP.md` stays the one place that says
what is next. When an effort is finished, its outcome goes into the document that owns the
fact and its `.scratch/` directory is deleted.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- One file per ticket: `.scratch/<feature-slug>/issues/<ticket-slug>.md`. The ticket slug
  is the ticket's name: a short kebab-case summary of its task, such as
  `config-refuses-unknown-keys`, unique across the whole project and never reused, so it
  names one ticket for good. Before taking one, check that no `.scratch/*/issues/` file and
  no commit subject (`git log --oneline --grep=<slug>`) has it
- A ticket's commit subject starts with `Ticket <ticket-slug>:`
- State is a `Status:` line near the top: `ready-for-agent` (an agent can do it offline),
  `ready-for-human` (it needs the hardware, the user's go or the user's decision), `done`
- Comments append at the bottom under `## Comments`

## When a skill says "publish to the issue tracker"

Create the file under `.scratch/<feature-slug>/`, creating the directory if needed.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user normally passes the path or the ticket slug.
