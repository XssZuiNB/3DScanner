# Keep CUDA_3D's code style

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

C++ and CUDA keep the style CUDA_3D was written in, through the rewrite too: the formatting
of `.clang-format` (Microsoft-based, 4-space indent, braces on their own line, 100 columns)
and snake_case for types, functions, variables and constants. This departs from the Google C++
guide that CODING_STANDARDS.md names; everything else in that guide applies. Code carried over
from CUDA_3D then needs no restyling, and old and new code read alike. Decided by the user,
2026-11-13.
