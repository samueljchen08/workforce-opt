#!/usr/bin/env bash
# Explicit personal-repository publication; no credentials are printed or stored here.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
gh auth status --hostname github.com >/dev/null
github_owner="$(gh api user --jq .login)"
if [[ "$github_owner" != "samueljchen08" ]]; then
  echo "Signed in as $github_owner; expected personal account samueljchen08. Switch accounts first." >&2
  exit 1
fi
target_repo="$github_owner/workforce-opt"
target_url="https://github.com/$target_repo.git"
if git remote get-url origin >/dev/null 2>&1; then
  current_origin="$(git remote get-url origin)"
  if [[ "$current_origin" != "$target_url" && "$current_origin" != "git@github.com:$target_repo.git" ]]; then
    echo "Origin points elsewhere ($current_origin); refusing to change it." >&2
    exit 1
  fi
else
  if gh repo view "$target_repo" >/dev/null 2>&1; then
    git remote add origin "$target_url"
  else
    gh repo create "$target_repo" --private --description "6.C571 robust help-desk ticket routing: LP, uncertainty, fairness and cross-training" --source "$project_root" --remote origin
  fi
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Uncommitted changes remain. Review and commit them before publishing." >&2
  exit 1
fi
git push -u origin HEAD:main
gh repo view "$target_repo" --json url --jq .url
