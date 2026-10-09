#!/bin/bash
# Uses the locked checkout Python when available; never starts local Ollama.
set -eu
umask 077
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
demo_python=${UMA_BCT_DEMO_PYTHON:-"$script_dir/../.venv/bin/python"}
if [ ! -x "$demo_python" ]; then
    demo_python=$(command -v python3) || { echo 'Python 3.12+ is required.' >&2; exit 2; }
fi
if [ "${UMA_BCT_DEMO_TIMED_CHILD:-}" = 1 ]; then
    exec "$demo_python" "$script_dir/demo_ec2.py" "$@"
fi
if [ "${1:-}" != run ]; then
    exec "$demo_python" "$script_dir/demo_ec2.py" "$@"
fi
for argument in "$@"; do
    case "$argument" in -h|--help) exec "$demo_python" "$script_dir/demo_ec2.py" "$@" ;; esac
done
# Wall timing starts outside Python and encloses its startup, readiness, SSH,
# inference, publication and verified retrieval. Formatting follows retrieval.
timing_dir=$(mktemp -d "${TMPDIR:-/tmp}/uma-bct-demo-time.XXXXXXXX")
trap 'rm -rf -- "$timing_dir"' EXIT
want_json=0
for argument in "$@"; do
    [ "$argument" != --json ] || want_json=1
done
exec 3>&2
set +e
LC_ALL=C UMA_BCT_DEMO_STDERR_FD=3 UMA_BCT_DEMO_TIMED_CHILD=1 /usr/bin/time -p \
    /bin/bash "$script_dir/demo-ec2.sh" "$@" --json \
    >"$timing_dir/summary.json" 2>"$timing_dir/time.txt"
status=$?
set -e
exec 3>&-
"$demo_python" "$script_dir/demo_ec2.py" _finish "$timing_dir/summary.json" \
    "$timing_dir/time.txt" "$want_json" "$status"
# The formatter retains nonzero CLI status, and reports timer failures as exit 2.
