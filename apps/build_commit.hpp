// SPDX-License-Identifier: 0BSD
/// @file build_commit.hpp
/// The tree's commit the programs are built from, for each scan's parameters (docs/SCAN_FILES.md,
/// "Parameters"): the one source of it for every program. apps/build_commit.cmake writes its
/// definition, build_commit.cpp, into the build directory at every build, and only when its text
/// changes, so that an unchanged tree relinks no program.

#pragma once

#include <string_view>

namespace gca
{
/// Gives the tree's commit the program was built from.
/// @return the commit's abbreviated hash, as `git rev-parse --short HEAD` gives it, followed by
///         "-dirty" when the tree held changes not committed, files untracked by git included,
///         at the build; "unknown" when the tree was built without its git repository (a copy
///         without .git, as tools/run_mutant.sh makes). A static string of hexadecimal digits,
///         "-dirty" and "unknown" alone, which a scan's parameters hold (docs/SCAN_FILES.md,
///         "Strings").
std::string_view build_commit() noexcept;
} // namespace gca
