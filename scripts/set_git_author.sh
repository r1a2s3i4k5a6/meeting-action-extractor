#!/usr/bin/env bash
# Usage: ./scripts/set_git_author.sh "Your Name" "you@example.com"
# Sets your identity and rewrites the author/committer of EVERY commit (run before the first push).
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 \"Your Name\" \"you@example.com\""; exit 1; }
git config user.name "$1"
git config user.email "$2"
git rebase -r --root --exec 'git commit --amend --reset-author --no-edit'
git log --format='%h %an <%ae> %s'
