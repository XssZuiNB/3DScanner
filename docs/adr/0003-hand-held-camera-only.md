# The camera is hand-held; there is no turntable

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

Every scan is one continuous pass of the hand-held camera around an object that stands still,
for tabletop objects and large objects alike. A large object cannot be turned, so the tracking
of a moving camera is needed anyway; one pipeline then serves both object classes, and the
support plane moves with the object and helps the tracking. A turntable would add a second
way of scanning in which the background stands still while the object moves, so the
background would have to be cut away before any registration, and it would add hardware.
Decided by the user, 2026-11-13.

## Consequences

* The operator has to move steadily: the web page shows the live model and the tracking state
  while the scan runs.
* The side an object stands on is not scanned. Merging a second scan of the turned object
  belongs to the mesh phase.
* The camera hangs on a cable to the Jetson, which stands still.
