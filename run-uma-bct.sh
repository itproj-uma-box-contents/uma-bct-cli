#!/bin/sh

set -eu
umask 077
PATH="/usr/bin:/bin:/usr/sbin:/sbin"
export PATH

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
INPUTS="$ROOT/inputs"
OUTPUTS="$ROOT/outputs"
RUNTIME="$ROOT/.uma-bct-runtime"
CONSTRAINTS="$ROOT/client-constraints.txt"
APP_ENV="$RUNTIME/app"
APP_PYTHON="$APP_ENV/bin/python"
APP_CLI="$APP_ENV/bin/uma-bct"
BUNDLED_UV_VERSION="0.12.9"
BUNDLED_UV_RELEASE_URL="https://releases.astral.sh/github/uv/releases/download/$BUNDLED_UV_VERSION"
OLLAMA_VERSION="0.30.8"
MINIMUM_OLLAMA_VERSION="$OLLAMA_VERSION"
OLLAMA_RELEASE_URL="https://github.com/ollama/ollama/releases/download/v$OLLAMA_VERSION"
MINIMUM_GLIBC_VERSION="2.28"
OLLAMA_LOG="$RUNTIME/ollama.log"
OLLAMA_PID=""

fail() {
    printf '\nUMA BCT could not continue: %s\n' "$1" >&2
    exit 2
}

stop_owned_ollama() {
    [ -n "$OLLAMA_PID" ] || return 0
    owned_pid=$OLLAMA_PID
    OLLAMA_PID=""
    if kill -0 "$owned_pid" 2>/dev/null; then
        kill "$owned_pid" 2>/dev/null || true
    fi
    wait "$owned_pid" 2>/dev/null || true
}

trap stop_owned_ollama EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

has_input_images() {
    for image in "$INPUTS"/*; do
        [ -f "$image" ] || continue
        case "$image" in
            *.[Jj][Pp][Gg]|*.[Jj][Pp][Ee][Gg]|*.[Pp][Nn][Gg]|*.[Tt][Ii][Ff]|*.[Tt][Ii][Ff][Ff]|*.[Ww][Ee][Bb][Pp])
                return 0
                ;;
        esac
    done
    return 1
}

has_private_data() {
    if ! private_entry=$(find "$INPUTS" "$OUTPUTS" -mindepth 1 -maxdepth 1 -print -quit); then
        fail "The inputs and outputs folders could not be inspected safely."
    fi
    [ -n "$private_entry" ]
}

open_folder() {
    [ "${UMA_BCT_NO_OPEN:-}" = "1" ] && return 0
    case "$(uname -s)" in
        Darwin) open "$1" >/dev/null 2>&1 || true ;;
        *) command -v xdg-open >/dev/null 2>&1 && xdg-open "$1" >/dev/null 2>&1 || true ;;
    esac
}

download_script() {
    url=$1
    destination=$2
    expected_sha256=$3
    has_private_data && fail "Private files appeared before online preparation finished."
    command -v curl >/dev/null 2>&1 || fail "curl is required for the first run."
    curl -q --proto '=https' --proto-redir '=https' --tlsv1.2 \
        -LsSf "$url" -o "$destination"
    if command -v shasum >/dev/null 2>&1; then
        actual_sha256=$(shasum -a 256 "$destination" | awk '{print $1}')
    elif command -v sha256sum >/dev/null 2>&1; then
        actual_sha256=$(sha256sum "$destination" | awk '{print $1}')
    else
        fail "A SHA-256 checksum tool is required for the first run."
    fi
    [ "$actual_sha256" = "$expected_sha256" ] || fail "Download checksum verification failed."
}

find_ollama() {
    for candidate in \
        "$RUNTIME/ollama/ollama" \
        "$RUNTIME/ollama/bin/ollama"
    do
        if [ -f "$candidate" ] && [ ! -L "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

find_llama_server() {
    ollama_directory=$(dirname -- "$1")
    case "$(uname -s)" in
        Darwin)
            candidates="$ollama_directory/llama-server
$RUNTIME/ollama/lib/ollama/llama-server"
            ;;
        Linux)
            candidates="$RUNTIME/ollama/lib/ollama/llama-server
$ollama_directory/../lib/ollama/llama-server"
            ;;
    esac
    printf '%s\n' "$candidates" | while IFS= read -r candidate; do
        if [ -f "$candidate" ] && [ ! -L "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            break
        fi
    done
}

select_ollama_port() {
    "$APP_PYTHON" -c '
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
'
}

ollama_is_healthy() {
    OLLAMA_HEALTH_PORT=$ollama_port "$APP_PYTHON" -c '
import http.client
import os

connection = http.client.HTTPConnection("127.0.0.1", int(os.environ["OLLAMA_HEALTH_PORT"]), timeout=0.5)
try:
    connection.request("GET", "/api/version")
    response = connection.getresponse()
    raise SystemExit(0 if response.status == 200 else 1)
finally:
    connection.close()
' >/dev/null 2>&1
}

start_owned_ollama() {
    [ ! -L "$OLLAMA_LOG" ] || fail "The Ollama runtime log must not be a symbolic link: $OLLAMA_LOG"
    [ ! -e "$OLLAMA_LOG" ] || [ -f "$OLLAMA_LOG" ] || fail "The Ollama runtime log must be a regular file."
    if ! : >"$OLLAMA_LOG"; then
        fail "The Ollama runtime log could not be prepared: $OLLAMA_LOG"
    fi
    for attempt in 1 2 3; do
        ollama_port=$(select_ollama_port || true)
        case "$ollama_port" in
            ''|*[!0-9]*) continue ;;
        esac
        [ "$ollama_port" -gt 0 ] && [ "$ollama_port" -le 65535 ] || continue
        OLLAMA_HOST="http://127.0.0.1:$ollama_port"
        export OLLAMA_HOST
        printf 'Starting the isolated Ollama service (attempt %s of 3)...\n' "$attempt"
        "$OLLAMA" serve >>"$OLLAMA_LOG" 2>&1 &
        OLLAMA_PID=$!

        deadline=$(($(date +%s) + 30))
        while [ "$(date +%s)" -lt "$deadline" ]; do
            if ollama_is_healthy && kill -0 "$OLLAMA_PID" 2>/dev/null; then
                sleep 1
                if [ "$(date +%s)" -lt "$deadline" ] && \
                    kill -0 "$OLLAMA_PID" 2>/dev/null && ollama_is_healthy
                then
                    return 0
                fi
            fi
            kill -0 "$OLLAMA_PID" 2>/dev/null || break
            sleep 1
        done
        stop_owned_ollama
    done
    fail "Ollama did not start after three attempts. Review the runtime log at $OLLAMA_LOG"
}

ollama_version_is_supported() {
    "$APP_PYTHON" -c '
import os
import re
import subprocess
import sys

environment = os.environ.copy()
environment["OLLAMA_HOST"] = "http://127.0.0.1:1"
try:
    result = subprocess.run(
        [sys.argv[1], "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=5,
        check=False,
        env=environment,
    )
except (OSError, subprocess.SubprocessError):
    raise SystemExit(1)
versions = re.findall(r"\d+\.\d+\.\d+", result.stdout)
found = tuple(map(int, versions[-1].split("."))) if versions else ()
required = tuple(map(int, sys.argv[2].split(".")))
raise SystemExit(0 if found >= required else 1)
' "$1" "$MINIMUM_OLLAMA_VERSION"
}

require_supported_glibc() {
    glibc_version=""
    if command -v getconf >/dev/null 2>&1; then
        glibc_version=$(
            { getconf GNU_LIBC_VERSION 2>/dev/null || true; } |
                awk '$1 == "glibc" { print $2; exit }'
        )
    fi

    if [ -z "$glibc_version" ]; then
        command -v ldd >/dev/null 2>&1 || \
            fail "UMA BCT requires glibc $MINIMUM_GLIBC_VERSION or newer; no supported glibc installation was detected."
        libc_details=$({ ldd --version 2>&1 || true; })
        case "$libc_details" in
            *[Mm][Uu][Ss][Ll]*)
                fail "UMA BCT requires glibc $MINIMUM_GLIBC_VERSION or newer; musl Linux is not supported."
                ;;
        esac
        glibc_version=$(
            printf '%s\n' "$libc_details" |
                awk '
                    tolower($0) ~ /(glibc|gnu libc)/ &&
                    match($0, /[0-9]+\.[0-9]+/) {
                        print substr($0, RSTART, RLENGTH)
                        exit
                    }
                '
        )
    fi

    [ -n "$glibc_version" ] || \
        fail "UMA BCT requires glibc $MINIMUM_GLIBC_VERSION or newer; no supported glibc installation was detected."
    if ! awk -v current="$glibc_version" -v minimum="$MINIMUM_GLIBC_VERSION" '
        BEGIN {
            split(current, found, ".")
            split(minimum, required, ".")
            if ((found[1] + 0) > (required[1] + 0)) exit 0
            if ((found[1] + 0) < (required[1] + 0)) exit 1
            exit !((found[2] + 0) >= (required[2] + 0))
        }
    '
    then
        fail "UMA BCT requires glibc $MINIMUM_GLIBC_VERSION or newer; found glibc $glibc_version."
    fi
}

install_ollama() {
    printf 'Installing the verified local Ollama service...\n'
    case "$(uname -s)-$(uname -m)" in
        Darwin-*)
            ollama_archive_name="ollama-darwin.tgz"
            ollama_archive_sha256="52acbca4e89c53db9abc586a22b5633fd101db293177264b9a0fe5d64a42a064"
            ollama_archive_type="gzip"
            ;;
        Linux-x86_64|Linux-amd64)
            ollama_archive_name="ollama-linux-amd64.tar.zst"
            ollama_archive_sha256="ffe2b2c2f2f5f5b30c081ec353c2e0bb2d9ead516064a8e22663b24b8fd8dca0"
            ollama_archive_type="zstd"
            ;;
        Linux-arm64|Linux-aarch64)
            ollama_archive_name="ollama-linux-arm64.tar.zst"
            ollama_archive_sha256="668a6f934b0b0455128bb4a76c9e50b9e5f274f9dc7710a066b7073e5bd36588"
            ollama_archive_type="zstd"
            ;;
        *) fail "No verified Ollama download is available for this computer." ;;
    esac
    ollama_archive="$RUNTIME/$ollama_archive_name"
    ollama_directory="$RUNTIME/ollama"
    download_script \
        "$OLLAMA_RELEASE_URL/$ollama_archive_name" \
        "$ollama_archive" \
        "$ollama_archive_sha256"
    rm -rf "$ollama_directory"
    mkdir -m 700 "$ollama_directory"
    if [ "$ollama_archive_type" = "gzip" ]; then
        tar -xzf "$ollama_archive" -C "$ollama_directory"
    else
        "$APP_PYTHON" "$ROOT/extract_ollama_archive.py" \
            "$ollama_archive" "$ollama_directory"
    fi
    rm -f "$ollama_archive"
    chmod -R go-rwx "$ollama_directory"
}

install_uv() {
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64)
            archive_name="uv-aarch64-apple-darwin.tar.gz"
            archive_sha256="301f72afaf54060f92da7016cb0115bd077f43a9c8e39c1d8170a0bac80fd398"
            ;;
        Darwin-x86_64)
            archive_name="uv-x86_64-apple-darwin.tar.gz"
            archive_sha256="e1ca175824f1056589ce9908f7631879ebc3c36535b5e63dc06510beb370b4c1"
            ;;
        Linux-arm64|Linux-aarch64)
            archive_name="uv-aarch64-unknown-linux-gnu.tar.gz"
            archive_sha256="c36fe17937ff6bd16dc42fc13854b5465999fcab2efe0af559381e945e3c6001"
            ;;
        Linux-x86_64|Linux-amd64)
            archive_name="uv-x86_64-unknown-linux-gnu.tar.gz"
            archive_sha256="ec7a99cd05e0cd7f80243f135ce1361c76835cb0ee60055d14d20eba8eba1460"
            ;;
        *) fail "No verified uv download is available for this computer." ;;
    esac
    printf 'Preparing the private local Python runtime...\n'
    uv_archive="$RUNTIME/$archive_name"
    download_script "$BUNDLED_UV_RELEASE_URL/$archive_name" "$uv_archive" "$archive_sha256"
    tar -xzf "$uv_archive" --strip-components 1 -C "$RUNTIME/bin"
    rm -f "$uv_archive"
}

case "$(uname -s)" in
    Darwin)
        macos_major=$(sw_vers -productVersion | cut -d. -f1)
        [ "$macos_major" -ge 14 ] || fail "UMA BCT requires macOS 14 or newer."
        ;;
    Linux)
        require_supported_glibc
        ;;
    *) fail "This launcher supports macOS and Linux. Use Run UMA BCT.cmd on Windows." ;;
esac
case "$(uname -m)" in
    x86_64|amd64|arm64|aarch64) ;;
    *) fail "This computer architecture is not supported by UMA BCT." ;;
esac

for directory in \
    "$INPUTS" "$OUTPUTS" "$RUNTIME" "$RUNTIME/bin" \
    "$RUNTIME/uv-cache" "$RUNTIME/python" "$APP_ENV" "$RUNTIME/ollama"
do
    [ ! -L "$directory" ] || fail "$directory must not be a symbolic link."
    [ ! -e "$directory" ] || [ -d "$directory" ] || fail "$directory must be a directory."
done
mkdir -p "$INPUTS" "$OUTPUTS" "$RUNTIME/bin"
chmod 700 "$RUNTIME" "$RUNTIME/bin"
private_data_present=0
has_private_data && private_data_present=1

wheel=""
wheel_count=0
for candidate in "$ROOT"/uma_bct_cli-*.whl; do
    [ -f "$candidate" ] || continue
    [ ! -L "$candidate" ] || fail "The supplied wheel must not be a symbolic link."
    wheel=$candidate
    wheel_count=$((wheel_count + 1))
done
[ "$wheel_count" -eq 1 ] || fail "The supplied folder must contain exactly one UMA BCT wheel."
[ -f "$CONSTRAINTS" ] && [ ! -L "$CONSTRAINTS" ] || fail "The supplied dependency lock is missing or redirected."

for variable in $(env | sed -n 's/^\(UV_[A-Za-z0-9_]*\)=.*/\1/p'); do
    unset "$variable"
done
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV PIP_INDEX_URL PIP_EXTRA_INDEX_URL 2>/dev/null || true
unset OLLAMA_HOST OLLAMA_DOWNLOAD_URL OLLAMA_INSTALL_DIR OLLAMA_NO_START 2>/dev/null || true
unset TAR_OPTIONS GZIP XZ_OPT ZSTD_CLEVEL ZSTD_NBTHREADS 2>/dev/null || true

UV="$RUNTIME/bin/uv"
[ ! -L "$UV" ] || fail "$UV must not be a symbolic link."
[ ! -e "$UV" ] || [ -f "$UV" ] || fail "$UV must be a regular file."
uv_version=$({ "$UV" --version 2>/dev/null || true; } | awk '{print $2}')
if [ "$uv_version" != "$BUNDLED_UV_VERSION" ]; then
    [ "$private_data_present" -eq 0 ] || \
        fail "Move private files out of inputs and outputs before repairing the local runtime."
    install_uv
fi
[ -x "$UV" ] || fail "The local Python runtime could not be installed."
uv_version=$({ "$UV" --version 2>/dev/null || true; } | awk '{print $2}')
[ "$uv_version" = "$BUNDLED_UV_VERSION" ] || fail "The local Python runtime has an unexpected version."

export UV_CACHE_DIR="$RUNTIME/uv-cache"
export UV_PYTHON_INSTALL_DIR="$RUNTIME/python"
export UV_DEFAULT_INDEX="https://pypi.org/simple"
export UV_NO_CONFIG=1
export OLLAMA_NO_CLOUD=1
unset UMA_BCT_ALLOW_REMOTE_OLLAMA UMA_BCT_ALLOW_CONTAINER_OLLAMA 2>/dev/null || true

if [ "$private_data_present" -eq 0 ]; then
    has_private_data && fail "Private files appeared before online preparation finished."
    printf 'Installing the verified application dependencies...\n'
    "$UV" venv --no-config --clear --managed-python --python 3.12 "$APP_ENV"
    "$UV" pip sync --no-config --require-hashes --python "$APP_PYTHON" "$CONSTRAINTS"
    "$UV" pip install --no-config --offline --no-deps --python "$APP_PYTHON" "$wheel"
else
    [ -x "$APP_PYTHON" ] || \
        fail "Move private files out of inputs and outputs before repairing the application environment."
    [ -f "$APP_CLI" ] && [ ! -L "$APP_CLI" ] || \
        fail "Move private files out of inputs and outputs before repairing the application environment."
    "$UV" pip install --no-config --offline --no-deps --reinstall \
        --python "$APP_PYTHON" "$wheel"
fi
[ -x "$APP_CLI" ] || fail "The application could not be installed."

OLLAMA=$(find_ollama || true)
LLAMA_SERVER=""
[ -z "$OLLAMA" ] || LLAMA_SERVER=$(find_llama_server "$OLLAMA")
if [ -z "$OLLAMA" ] || ! ollama_version_is_supported "$OLLAMA" || [ -z "$LLAMA_SERVER" ]; then
    [ "$private_data_present" -eq 0 ] || fail \
        "The local Ollama runtime is incomplete or unsupported; both ollama and llama-server are required. Move private files out of inputs and outputs, then run this launcher again to repair it. Runtime log: $OLLAMA_LOG"
    install_ollama
    OLLAMA=$(find_ollama || true)
    LLAMA_SERVER=""
    [ -z "$OLLAMA" ] || LLAMA_SERVER=$(find_llama_server "$OLLAMA")
fi
[ -n "$OLLAMA" ] || fail "Ollama could not be installed. Runtime log: $OLLAMA_LOG"
ollama_version_is_supported "$OLLAMA" || fail "Ollama is too old and could not be updated. Runtime log: $OLLAMA_LOG"
[ -n "$LLAMA_SERVER" ] || fail "Ollama's llama-server companion is missing. Runtime log: $OLLAMA_LOG"

start_owned_ollama

cd "$ROOT"
if [ "$private_data_present" -eq 0 ]; then
    has_private_data && fail "Private files appeared before online preparation finished."
    printf 'Checking the application and model. The first model download is large...\n'
    if ! "$APP_CLI" setup >"$RUNTIME/setup.json"
    then
        fail "Automatic setup failed. See the message above and run this launcher again."
    fi
fi

if ! has_input_images; then
    printf 'Add JPG, PNG, TIF, or WEBP photos to the folder that just opened.\n'
    open_folder "$INPUTS"
    if [ -t 0 ]; then
        printf 'Press Enter after the photos have finished copying: '
        read -r _answer
    fi
fi
has_input_images || fail "No supported photos were found in $INPUTS"

printf 'Transcribing the photos...\n'
set +e
"$APP_CLI" --force
status=$?
set -e

if [ "$status" -eq 0 ]; then
    open_folder "$OUTPUTS"
    printf '\nFinished. Review %s/catalogue.csv\n' "$OUTPUTS"
elif [ "$status" -eq 1 ]; then
    open_folder "$OUTPUTS"
    printf '\nProcessing was incomplete. Review the newest run manifest in %s.\n' "$OUTPUTS" >&2
    printf 'The catalogue may be partial, absent, or retained from the previous run.\n' >&2
fi
exit "$status"
