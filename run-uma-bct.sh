#!/bin/sh

# Keep new client files private and resolve bootstrap commands from trusted
# system locations before the launcher touches photos or downloads.
set -eu
umask 077
PATH="/usr/bin:/bin:/usr/sbin:/sbin"
export PATH

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
FRIENDLY_INPUTS="$ROOT/Add Photos Here"
FRIENDLY_OUTPUTS="$ROOT/Results"
LEGACY_INPUTS="$ROOT/inputs"
LEGACY_OUTPUTS="$ROOT/outputs"
INPUTS=""
OUTPUTS=""
RUNTIME="$ROOT/.uma-bct-runtime"
CONSTRAINTS="$ROOT/client-constraints.txt"
APP_ENV="$RUNTIME/app"
APP_PYTHON="$APP_ENV/bin/python"
APP_CLI="$APP_ENV/bin/uma-bct"
BUNDLED_UV_VERSION="0.12.9"
BUNDLED_UV_RELEASE_URL="https://releases.astral.sh/github/uv/releases/download/$BUNDLED_UV_VERSION"
OLLAMA_VERSION="0.32.9"
MINIMUM_OLLAMA_VERSION="$OLLAMA_VERSION"
OLLAMA_RELEASE_URL="https://github.com/ollama/ollama/releases/download/v$OLLAMA_VERSION"
MINIMUM_GLIBC_VERSION="2.28"
OLLAMA_LOG="$RUNTIME/ollama.log"
OLLAMA_PID=""
APP_PID=""
LAUNCH_LOCK="$ROOT/.uma-bct-launcher.lock"
LAUNCH_LOCK_HELD=0
LAUNCH_LOCK_IDENTITY=""

fail() {
    printf '\nUMA BCT could not continue: %s\n' "$1" >&2
    exit 2
}

directory_has_data() {
    directory=$1
    [ ! -L "$directory" ] || fail "$directory must not be a symbolic link."
    [ -e "$directory" ] || return 1
    [ -d "$directory" ] || fail "$directory must be a directory."
    [ -r "$directory" ] && [ -x "$directory" ] || fail "$directory could not be inspected safely."
    for entry in "$directory"/* "$directory"/.[!.]* "$directory"/..?*; do
        [ -e "$entry" ] || [ -L "$entry" ] || continue
        if [ "$(basename -- "$entry")" = ".uma-bct.lock" ] && verified_results_lock "$entry"; then
            continue
        fi
        if [ "$(basename -- "$entry")" = ".uma-bct-settings.json" ] && verified_settings_file "$entry"; then
            continue
        fi
        return 0
    done
    return 1
}

verified_settings_file() {
    settings_path=$1
    [ -f "$settings_path" ] && [ ! -L "$settings_path" ] && [ -r "$settings_path" ] || return 1
    link_count=$({ stat -c '%h' "$settings_path" 2>/dev/null || stat -f '%l' "$settings_path" 2>/dev/null || true; })
    [ "$link_count" = "1" ] || return 1
    size=$(wc -c <"$settings_path" | awk '{print $1}') || return 1
    [ "$size" -le 4096 ] || return 1
    LC_ALL=C tr -d '[:space:]' <"$settings_path" | LC_ALL=C grep -Exq '\{"verification":"(ask|always|never)"\}'
}

verified_results_lock() {
    lock_path=$1
    [ -f "$lock_path" ] && [ ! -L "$lock_path" ] && [ -r "$lock_path" ] || return 1
    link_count=$({ stat -c '%h' "$lock_path" 2>/dev/null || stat -f '%l' "$lock_path" 2>/dev/null || true; })
    [ "$link_count" = "1" ] || return 1
    line_count=$(wc -l <"$lock_path" | awk '{print $1}') || return 1
    [ "$line_count" = "1" ] || return 1
    lock_content=$(LC_ALL=C sed -n '1,$p' "$lock_path" 2>/dev/null) || return 1
    case "$lock_content" in
        *"
"*) return 1 ;;
    esac
    printf '%s\n' "$lock_content" | LC_ALL=C grep -Exq '\{"schema_version": 1, "pid": [1-9][0-9]*\}'
}

path_identity() {
    stat -c '%d:%i' "$1" 2>/dev/null || stat -f '%d:%i' "$1" 2>/dev/null
}

read_launcher_owner() {
    owner_path=$1
    [ -f "$owner_path" ] && [ ! -L "$owner_path" ] && [ -r "$owner_path" ] || return 1
    owner_links=$({ stat -c '%h' "$owner_path" 2>/dev/null || stat -f '%l' "$owner_path" 2>/dev/null || true; })
    [ "$owner_links" = "1" ] || return 1
    owner_value=$(LC_ALL=C sed -n '1,$p' "$owner_path" 2>/dev/null) || return 1
    case "$owner_value" in
        ''|0*|*[!0-9]*) return 1 ;;
    esac
    [ "$(wc -l <"$owner_path" | awk '{print $1}')" = "1" ] || return 1
    [ "$(wc -c <"$owner_path" | awk '{print $1}')" -eq "$((${#owner_value} + 1))" ] || return 1
    printf '%s\n' "$owner_value"
}

select_data_directories() {
    # Existing legacy folders stay in place. Mixed layouts need a person to
    # choose the source of truth before this launcher can safely resume.
    friendly_data=0
    legacy_data=0
    if directory_has_data "$FRIENDLY_INPUTS" || directory_has_data "$FRIENDLY_OUTPUTS"; then
        friendly_data=1
    fi
    if directory_has_data "$LEGACY_INPUTS" || directory_has_data "$LEGACY_OUTPUTS"; then
        legacy_data=1
    fi
    if [ "$friendly_data" -eq 1 ] && [ "$legacy_data" -eq 1 ]; then
        fail "Private data exists in both '$FRIENDLY_INPUTS' / '$FRIENDLY_OUTPUTS' and '$LEGACY_INPUTS' / '$LEGACY_OUTPUTS'. Keep only one layout; files were not merged or moved."
    fi
    if [ "$legacy_data" -eq 1 ]; then
        INPUTS=$LEGACY_INPUTS
        OUTPUTS=$LEGACY_OUTPUTS
    else
        INPUTS=$FRIENDLY_INPUTS
        OUTPUTS=$FRIENDLY_OUTPUTS
    fi
}

reserve_bundle() {
    # The directory reservation covers setup and transcription together.
    # Recover a dead owner only after validating and rechecking its identity.
    if ! mkdir -m 700 "$LAUNCH_LOCK" 2>/dev/null; then
        [ -d "$LAUNCH_LOCK" ] && [ ! -L "$LAUNCH_LOCK" ] || \
            fail "The launcher reservation is malformed and was not recovered: $LAUNCH_LOCK"
        stale_lock_identity=$(path_identity "$LAUNCH_LOCK") || \
            fail "The launcher reservation identity could not be inspected: $LAUNCH_LOCK"
        owner_file="$LAUNCH_LOCK/pid"
        entry_count=$(find "$LAUNCH_LOCK" -mindepth 1 -maxdepth 1 -print 2>/dev/null | wc -l | awk '{print $1}') || \
            fail "The launcher reservation could not be inspected: $LAUNCH_LOCK"
        [ "$entry_count" = "1" ] || fail "The launcher reservation is malformed and was not recovered: $LAUNCH_LOCK"
        owner_pid=$(read_launcher_owner "$owner_file") || \
            fail "The launcher reservation is malformed and was not recovered: $LAUNCH_LOCK"
        stale_owner_identity=$(path_identity "$owner_file") || \
            fail "The launcher reservation owner identity could not be inspected: $owner_file"
        if kill -0 "$owner_pid" 2>/dev/null || ps -p "$owner_pid" >/dev/null 2>&1; then
            fail "This extracted UMA BCT folder is already running. Wait for the other launcher to finish: $ROOT"
        fi
        recovery="$ROOT/.uma-bct-lock-release-launcher-recovery-$$.tmp"
        [ ! -e "$recovery" ] && [ ! -L "$recovery" ] || \
            fail "The launcher reservation recovery path already exists: $recovery"
        if ! mv "$LAUNCH_LOCK" "$recovery" 2>/dev/null; then
            fail "The launcher reservation changed during recovery: $LAUNCH_LOCK"
        fi
        moved_lock_identity=$(path_identity "$recovery" 2>/dev/null || true)
        moved_owner_identity=$(path_identity "$recovery/pid" 2>/dev/null || true)
        moved_owner=$(read_launcher_owner "$recovery/pid" 2>/dev/null || true)
        if [ "$moved_lock_identity" != "$stale_lock_identity" ] || \
            [ "$moved_owner_identity" != "$stale_owner_identity" ] || \
            [ "$moved_owner" != "$owner_pid" ]
        then
            if [ ! -e "$LAUNCH_LOCK" ] && [ ! -L "$LAUNCH_LOCK" ]; then
                mv "$recovery" "$LAUNCH_LOCK" 2>/dev/null || true
            fi
            fail "The launcher reservation changed during recovery and was not removed: $LAUNCH_LOCK"
        fi
        rm -f "$recovery/pid"
        rmdir "$recovery" || fail "The stale launcher reservation recovery was not empty: $recovery"
        if ! mkdir -m 700 "$LAUNCH_LOCK" 2>/dev/null; then
            fail "Another launcher won the reservation recovery race: $ROOT"
        fi
    fi
    LAUNCH_LOCK_IDENTITY=$(path_identity "$LAUNCH_LOCK") || \
        fail "The new launcher reservation identity could not be inspected: $LAUNCH_LOCK"
    LAUNCH_LOCK_HELD=1
    owner_temporary="$LAUNCH_LOCK/.pid-$$.tmp"
    if ! printf '%s\n' "$$" >"$owner_temporary"; then
        fail "The launcher reservation owner could not be written: $LAUNCH_LOCK"
    fi
    mv "$owner_temporary" "$LAUNCH_LOCK/pid" || \
        fail "The launcher reservation owner could not be published: $LAUNCH_LOCK"
}

release_bundle() {
    # Remove only the reservation this process created; a swapped entry may
    # belong to another launcher and must remain available for recovery.
    [ "$LAUNCH_LOCK_HELD" -eq 1 ] || return 0
    LAUNCH_LOCK_HELD=0
    [ -n "$LAUNCH_LOCK_IDENTITY" ] || return 0
    [ -d "$LAUNCH_LOCK" ] && [ ! -L "$LAUNCH_LOCK" ] || return 0
    current_identity=$(path_identity "$LAUNCH_LOCK" 2>/dev/null || true)
    [ "$current_identity" = "$LAUNCH_LOCK_IDENTITY" ] || return 0
    release_path="$ROOT/.uma-bct-lock-release-launcher-$$.tmp"
    [ ! -e "$release_path" ] && [ ! -L "$release_path" ] || return 0
    mv "$LAUNCH_LOCK" "$release_path" 2>/dev/null || return 0
    moved_identity=$(path_identity "$release_path" 2>/dev/null || true)
    if [ "$moved_identity" != "$LAUNCH_LOCK_IDENTITY" ]; then
        [ -e "$LAUNCH_LOCK" ] || [ -L "$LAUNCH_LOCK" ] || \
            mv "$release_path" "$LAUNCH_LOCK" 2>/dev/null || true
        return 0
    fi
    rm -f "$release_path/pid" 2>/dev/null || true
    rm -f "$release_path/.pid-$$.tmp" 2>/dev/null || true
    rmdir "$release_path" 2>/dev/null || true
    LAUNCH_LOCK_IDENTITY=""
}

process_is_running() {
    checked_pid=$1
    kill -0 "$checked_pid" 2>/dev/null || return 1
    process_state=$(ps -o stat= -p "$checked_pid" 2>/dev/null | awk 'NR == 1 { print $1 }')
    case "$process_state" in
        ''|Z*) return 1 ;;
    esac
    return 0
}

stop_process_bounded() {
    owned_pid=$1
    signal_name=$2
    process_is_running "$owned_pid" || return 0
    kill -"$signal_name" "$owned_pid" 2>/dev/null || true
    countdown=50
    while [ "$countdown" -gt 0 ] && process_is_running "$owned_pid"; do
        sleep 0.1
        countdown=$((countdown - 1))
    done
    if process_is_running "$owned_pid"; then
        kill -KILL "$owned_pid" 2>/dev/null || true
    fi
}

stop_owned_ollama() {
    [ -n "$OLLAMA_PID" ] || return 0
    owned_pid=$OLLAMA_PID
    OLLAMA_PID=""
    stop_process_bounded "$owned_pid" TERM
    wait "$owned_pid" 2>/dev/null || true
}

stop_foreground_cli() {
    [ -n "$APP_PID" ] || return 0
    owned_pid=$APP_PID
    APP_PID=""
    stop_process_bounded "$owned_pid" TERM
    wait "$owned_pid" 2>/dev/null || true
}

cleanup() {
    stop_foreground_cli
    stop_owned_ollama
    release_bundle
}

forward_interrupt() {
    # Let the foreground CLI checkpoint the interruption before the exit trap
    # stops the Ollama child owned by this extracted bundle.
    exit_code=$1
    signal_name=$2
    if [ -n "$APP_PID" ]; then
        owned_pid=$APP_PID
        kill -"$signal_name" "$owned_pid" 2>/dev/null || true
        countdown=50
        while [ "$countdown" -gt 0 ] && process_is_running "$owned_pid"; do
            sleep 0.1
            countdown=$((countdown - 1))
        done
    fi
    exit "$exit_code"
}

trap cleanup EXIT
trap 'forward_interrupt 129 HUP' HUP
trap 'forward_interrupt 130 INT' INT
trap 'forward_interrupt 143 TERM' TERM

has_input_images() {
    for image in "$INPUTS"/*; do
        [ -f "$image" ] || continue
        case "$image" in
            *.[Aa][Vv][Ii][Ff]|*.[Bb][Mm][Pp]|*.[Dd][Nn][Gg]|*.[Gg][Ii][Ff]|*.[Hh][Ee][Ii][Cc]|*.[Hh][Ee][Ii][Ff]|*.[Jj][Pp][Gg]|*.[Jj][Pp][Ee][Gg]|*.[Pp][Nn][Gg]|*.[Tt][Ii][Ff]|*.[Tt][Ii][Ff][Ff]|*.[Ww][Ee][Bb][Pp])
                return 0
                ;;
        esac
    done
    return 1
}

has_private_data() {
    directory_has_data "$FRIENDLY_INPUTS" || directory_has_data "$FRIENDLY_OUTPUTS" || \
        directory_has_data "$LEGACY_INPUTS" || directory_has_data "$LEGACY_OUTPUTS"
}

open_folder() {
    [ "${UMA_BCT_NO_OPEN:-}" = "1" ] && return 0
    case "$(uname -s)" in
        Darwin)
            if open -a Finder "$1" >/dev/null 2>&1; then return 0; fi
            if open "$1" >/dev/null 2>&1; then return 0; fi
            ;;
        *)
            if command -v xdg-open >/dev/null 2>&1 && xdg-open "$1" >/dev/null 2>&1; then return 0; fi
            ;;
    esac
    printf 'Could not open the folder automatically. Open it in your file manager: %s\n' "$1" >&2
}

download_script() {
    # Online preparation happens before photos are added. Each download is
    # accepted only after its pinned SHA-256 matches the expected release.
    url=$1
    destination=$2
    expected_sha256=$3
    has_private_data && fail "Private files appeared before online preparation finished."
    command -v curl >/dev/null 2>&1 || fail "curl is required for the first run."
    curl -q --proto '=https' --proto-redir '=https' --tlsv1.2 \
        -LsSf "$url" -o "$destination" &
    APP_PID=$!
    if ! wait "$APP_PID"; then
        APP_PID=""
        fail "The verified download did not complete."
    fi
    APP_PID=""
    if command -v shasum >/dev/null 2>&1; then
        actual_sha256=$(shasum -a 256 "$destination" | awk '{print $1}')
    elif command -v sha256sum >/dev/null 2>&1; then
        actual_sha256=$(sha256sum "$destination" | awk '{print $1}')
    else
        fail "A SHA-256 checksum tool is required for the first run."
    fi
    [ "$actual_sha256" = "$expected_sha256" ] || fail "Download checksum verification failed."
}

find_ollama_in() {
    search_root=$1
    for candidate in \
        "$search_root/ollama" \
        "$search_root/bin/ollama"
    do
        if [ -f "$candidate" ] && [ ! -L "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

find_ollama() {
    find_ollama_in "$RUNTIME/ollama"
}

find_llama_server() {
    ollama_directory=$(dirname -- "$1")
    search_root=${2:-"$RUNTIME/ollama"}
    case "$(uname -s)" in
        Darwin)
            candidates="$ollama_directory/llama-server
$search_root/lib/ollama/llama-server"
            ;;
        Linux)
            candidates="$search_root/lib/ollama/llama-server
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
    # Give each launch a fresh loopback port and confirm the same child stays
    # healthy twice before sending any private image through its API.
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
            ollama_archive_sha256="17a5b096d4515d00a6415012db847a2b353b389ed7ab33d025e3b98c2f05b49c"
            ollama_archive_type="gzip"
            ;;
        Linux-x86_64|Linux-amd64)
            ollama_archive_name="ollama-linux-amd64.tar.zst"
            ollama_archive_sha256="5d747a43369f61e38f20b5a39fcc5c90647e562cdc61e2e56034f1c5b113d540"
            ollama_archive_type="zstd"
            ;;
        Linux-arm64|Linux-aarch64)
            ollama_archive_name="ollama-linux-arm64.tar.zst"
            ollama_archive_sha256="79617139521db251c716d6229505d7530171ff4d68e476d0c22dabc15726a237"
            ollama_archive_type="zstd"
            ;;
        *) fail "No verified Ollama download is available for this computer." ;;
    esac
    ollama_archive="$RUNTIME/$ollama_archive_name"
    ollama_directory="$RUNTIME/ollama"
    ollama_staging="$RUNTIME/ollama-install"
    ollama_previous="$RUNTIME/ollama-previous"
    download_script \
        "$OLLAMA_RELEASE_URL/$ollama_archive_name" \
        "$ollama_archive" \
        "$ollama_archive_sha256"
    [ ! -L "$ollama_staging" ] || fail "The Ollama staging path must not be a symbolic link."
    [ ! -L "$ollama_previous" ] || fail "The Ollama recovery path must not be a symbolic link."
    rm -rf "$ollama_staging"
    mkdir -m 700 "$ollama_staging"
    if [ "$ollama_archive_type" = "gzip" ]; then
        tar -xzf "$ollama_archive" -C "$ollama_staging" &
    else
        "$APP_PYTHON" "$ROOT/extract_ollama_archive.py" \
            "$ollama_archive" "$ollama_staging" &
    fi
    APP_PID=$!
    if ! wait "$APP_PID"; then
        APP_PID=""
        fail "The verified Ollama archive could not be extracted."
    fi
    APP_PID=""
    rm -f "$ollama_archive"
    chmod -R go-rwx "$ollama_staging"
    staged_ollama=$(find_ollama_in "$ollama_staging" || true)
    staged_llama_server=""
    [ -z "$staged_ollama" ] || staged_llama_server=$(find_llama_server "$staged_ollama" "$ollama_staging")
    [ -n "$staged_ollama" ] && [ -n "$staged_llama_server" ] && \
        ollama_version_is_supported "$staged_ollama" || \
        fail "The staged Ollama runtime is incomplete or unsupported. Runtime log: $OLLAMA_LOG"

    # Promote the verified runtime from staging. Keep the previous directory
    # until the new one is in place so a failed promotion can be reversed.
    rm -rf "$ollama_previous"
    if [ -d "$ollama_directory" ]; then
        mv "$ollama_directory" "$ollama_previous"
    fi
    if mv "$ollama_staging" "$ollama_directory"; then
        rm -rf "$ollama_previous"
    else
        [ ! -d "$ollama_previous" ] || mv "$ollama_previous" "$ollama_directory" || true
        fail "The staged Ollama runtime could not be installed. Runtime log: $OLLAMA_LOG"
    fi
}

recover_ollama_promotion() {
    ollama_directory="$RUNTIME/ollama"
    ollama_previous="$RUNTIME/ollama-previous"
    [ ! -L "$ollama_previous" ] || fail "The Ollama recovery path must not be a symbolic link."
    if [ ! -e "$ollama_directory" ] && [ -d "$ollama_previous" ]; then
        mv "$ollama_previous" "$ollama_directory"
    fi
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

select_data_directories
reserve_bundle

# Check the entire private workspace before creating or repairing a runtime;
# a linked directory could otherwise send files outside this bundle.
for directory in \
    "$INPUTS" "$OUTPUTS" "$RUNTIME" "$RUNTIME/bin" \
    "$RUNTIME/uv-cache" "$RUNTIME/python" "$APP_ENV" "$RUNTIME/ollama" \
    "$RUNTIME/ollama-install" "$RUNTIME/ollama-previous"
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

# Ignore inherited installer settings and endpoint overrides. The bundle
# selects its own pinned tools, local cache, and isolated Ollama service.
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
        fail "Move private files out of '$INPUTS' and '$OUTPUTS' before repairing the local runtime."
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
    # Finish all online dependencies before inviting the client to add photos.
    has_private_data && fail "Private files appeared before online preparation finished."
    printf 'Installing the verified application dependencies...\n'
    "$UV" venv --no-config --clear --managed-python --python 3.12 "$APP_ENV"
    "$UV" pip sync --no-config --require-hashes --python "$APP_PYTHON" "$CONSTRAINTS"
    "$UV" pip install --no-config --offline --no-deps --python "$APP_PYTHON" "$wheel"
else
    [ -x "$APP_PYTHON" ] || \
        fail "Move private files out of '$INPUTS' and '$OUTPUTS' before repairing the application environment."
    [ -f "$APP_CLI" ] && [ ! -L "$APP_CLI" ] || \
        fail "Move private files out of '$INPUTS' and '$OUTPUTS' before repairing the application environment."
    "$UV" pip install --no-config --offline --no-deps --reinstall \
        --python "$APP_PYTHON" "$wheel"
fi
[ -x "$APP_CLI" ] || fail "The application could not be installed."

recover_ollama_promotion
OLLAMA=$(find_ollama || true)
LLAMA_SERVER=""
[ -z "$OLLAMA" ] || LLAMA_SERVER=$(find_llama_server "$OLLAMA")
if [ -z "$OLLAMA" ] || ! ollama_version_is_supported "$OLLAMA" || [ -z "$LLAMA_SERVER" ]; then
    [ "$private_data_present" -eq 0 ] || fail \
        "The local Ollama runtime is incomplete or unsupported; both ollama and llama-server are required. Move private files out of '$INPUTS' and '$OUTPUTS', then run this launcher again to repair it. Runtime log: $OLLAMA_LOG"
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
    "$APP_CLI" setup --path-to-inputs "$INPUTS" --results-directory "$OUTPUTS" >"$RUNTIME/setup.json" &
    APP_PID=$!
    if ! wait "$APP_PID"; then
        APP_PID=""
        fail "Automatic setup failed. See the message above and run this launcher again."
    fi
    APP_PID=""
fi

if ! has_input_images; then
    printf 'Add supported photo files (including JPG, PNG, HEIC, or DNG) to %s.\n' "$INPUTS"
    open_folder "$INPUTS"
    if [ -t 0 ]; then
        printf 'Press Enter after the photos have finished copying: '
        read -r _answer
    fi
fi
has_input_images || fail "No supported photos were found in $INPUTS"

set +e
# Leave the CLI attached to this terminal so progress stays on stderr and its
# final readable report appears here before the Results folder opens.
# Explicit stdin redirection keeps a background child attached to the terminal.
"$APP_CLI" --path-to-inputs "$INPUTS" --results-directory "$OUTPUTS" <&0 &
APP_PID=$!
wait "$APP_PID"
status=$?
APP_PID=""
set -e

if [ "$status" -eq 0 ] || [ "$status" -eq 1 ]; then
    open_folder "$OUTPUTS"
fi
exit "$status"
