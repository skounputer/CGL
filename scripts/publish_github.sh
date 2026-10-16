#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Creates a NEW public repository in the already-authorized account and pushes
# the clean local main branch. Existing remote repositories are never replaced.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
gh_bin="${GH_BIN:-gh}"
git_bin="${GIT_BIN:-git}"
repo=skounputer/CGL

"$gh_bin" auth status --hostname github.com
account="$("$gh_bin" api user --jq .login)"
if [[ "$account" != skounputer ]]; then
    printf 'Expected GitHub account skounputer; authenticated as %s.\n' "$account" >&2
    exit 1
fi
if [[ -n "$("$git_bin" status --porcelain)" ]]; then
    printf 'Commit or remove local changes before publishing.\n' >&2
    exit 1
fi
if [[ "$("$git_bin" branch --show-current)" != main ]]; then
    printf 'Publish from the main branch.\n' >&2
    exit 1
fi
expected="https://github.com/$repo.git"
existing="$("$git_bin" remote get-url origin 2>/dev/null || true)"
if [[ -n "$existing" && "$existing" != "$expected" ]]; then
    printf 'origin points to an unexpected repository; stopping.\n' >&2
    exit 1
fi
# gh exits if this repository already exists, avoiding accidental overwrite.
"$gh_bin" repo create "$repo" --public \
    --description 'CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning — research code'
if [[ -z "$existing" ]]; then "$git_bin" remote add origin "$expected"; fi
gh_path="$(command -v "$gh_bin")"
printf -v credential_helper '!%q auth git-credential' "$gh_path"
"$git_bin" -c credential.helper= -c "credential.helper=$credential_helper" push -u origin main
"$gh_bin" repo view "$repo" --json url,isPrivate --jq '{url, isPrivate}'
