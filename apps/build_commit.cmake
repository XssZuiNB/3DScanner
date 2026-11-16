# SPDX-License-Identifier: 0BSD
#
# apps/build_commit.cmake - writes build_commit.cpp, the definition of build_commit()
# (apps/build_commit.hpp): the tree's commit the programs are built from. Run at every build by
# apps/CMakeLists.txt as `cmake -DSOURCE_DIR=<tree> -DOUTPUT=<file> -P build_commit.cmake`; the
# file is replaced only when its text changes, so that an unchanged tree relinks no program.
# The commit is `git rev-parse --short HEAD`, with "-dirty" when `git status --porcelain` lists
# anything (a change not committed, or a file untracked by git); "unknown" when SOURCE_DIR is not
# the top of a git working tree (a copy without .git) or git is missing.

set(commit "unknown")
find_program(GCA_GIT git)
if(GCA_GIT)
    execute_process(
        COMMAND "${GCA_GIT}" -C "${SOURCE_DIR}" rev-parse --show-toplevel
        OUTPUT_VARIABLE top
        RESULT_VARIABLE top_failed
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    # A copy of the tree inside another repository would otherwise give that repository's commit.
    if(top_failed EQUAL 0)
        file(REAL_PATH "${top}" top)
    endif()
    file(REAL_PATH "${SOURCE_DIR}" source)
    if(top_failed EQUAL 0 AND top STREQUAL source)
        execute_process(
            COMMAND "${GCA_GIT}" -C "${SOURCE_DIR}" rev-parse --short HEAD
            OUTPUT_VARIABLE head
            RESULT_VARIABLE head_failed
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        execute_process(
            COMMAND "${GCA_GIT}" -C "${SOURCE_DIR}" status --porcelain
            OUTPUT_VARIABLE changes
            RESULT_VARIABLE status_failed
            ERROR_QUIET
        )
        if(head_failed EQUAL 0 AND status_failed EQUAL 0 AND head MATCHES "^[0-9a-f]+$")
            set(commit "${head}")
            if(NOT changes STREQUAL "")
                string(APPEND commit "-dirty")
            endif()
        endif()
    endif()
endif()

file(WRITE "${OUTPUT}.new" "// SPDX-License-Identifier: 0BSD
/// @file build_commit.cpp
/// Written by apps/build_commit.cmake at the build: the tree's commit (see build_commit.hpp).

#include \"build_commit.hpp\"

namespace gca
{
std::string_view build_commit() noexcept
{
    return \"${commit}\";
}
} // namespace gca
")
file(COPY_FILE "${OUTPUT}.new" "${OUTPUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUTPUT}.new")
