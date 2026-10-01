#!/usr/bin/env bash
set -euo pipefail

# Fail open to full CI when a comparison is unavailable or has no changed paths.
if [[ $# -ne 3 ]]; then
  echo false
  exit 0
fi

paths_file="$(mktemp)"
trap 'rm -f "$paths_file"' EXIT

case "$1" in
  pull_request)
    if ! git diff --no-renames --name-only -z "$2...$3" -- > "$paths_file"; then
      echo false
      exit 0
    fi
    ;;
  push)
    if ! git diff --no-renames --name-only -z "$2" "$3" -- > "$paths_file"; then
      echo false
      exit 0
    fi
    ;;
  *)
    echo false
    exit 0
    ;;
esac

if [[ ! -s "$paths_file" ]]; then
  echo false
  exit 0
fi

while IFS= read -r -d '' path; do
  case "$path" in
    docs/*) continue ;;
  esac
  shopt -s nocasematch
  case "$path" in
    *.md|*.markdown) ;;
    *)
      echo false
      exit 0
      ;;
  esac
  shopt -u nocasematch
done < "$paths_file"

echo true
