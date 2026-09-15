#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
IMAGE=${PHYRC_IMAGE:-phyrc-2027:isaac-6.0.1}
COMMAND=${1:-help}
if [[ "$COMMAND" == gui ]]; then
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --evaluate)
                [[ "${2:-}" == 0 || "${2:-}" == 1 ]] || { printf -- '--evaluate requires 0 or 1\n' >&2; exit 2; }
                export STRETCH4_AUTO_EVALUATE="$2"
                shift 2;;
            --training-render)
                [[ "${2:-}" == deferred || "${2:-}" == live ]] || { printf -- '--training-render requires deferred or live\n' >&2; exit 2; }
                export STRETCH4_TRAINING_RENDER="$2"
                shift 2;;
            --training-record)
                [[ "${2:-}" == 0 || "${2:-}" == 1 ]] || { printf -- '--training-record requires 0 or 1\n' >&2; exit 2; }
                export STRETCH4_TRAINING_RECORD="$2"
                shift 2;;
            --full-record)
                [[ "${2:-}" == 0 || "${2:-}" == 1 ]] || { printf -- '--full-record requires 0 or 1\n' >&2; exit 2; }
                export STRETCH4_FULL_RECORD="$2"
                shift 2;;
            --no-randomization) export STRETCH4_RANDOMIZE=0; shift;;
            *) printf 'Unknown gui option: %s\n' "$1" >&2; exit 2;;
        esac
    done
    set -- gui
fi
if [[ "${2:-}" == --no-randomization ]]; then
    case "$COMMAND" in
        gui|smoke|python)
            export STRETCH4_RANDOMIZE=0
            set -- "$1" "${@:3}";;
        *) printf -- '--no-randomization requires gui, smoke, or python.\n' >&2; exit 2;;
    esac
fi
if [[ "$COMMAND" == help ]]; then
    printf 'Usage: ./run.sh {doctor|build|prepare|gui|replay RECORDING [OPTIONS...]|smoke|python SCRIPT [ARGS...]|cpu SCRIPT [ARGS...]}\n'
    printf 'Use ./run.sh gui --no-randomization for fixed human and shirt placement.\n'
    printf 'Use ./run.sh gui --full-record 1 for a replay/evaluation state archive only (no training images or HDF5).\n'
    printf 'Use ./run.sh gui --training-record 1 for synchronized RGBD/action HDF5 plus full recording.\n'
    printf 'Use ./run.sh gui --evaluate 1 to score automatically from startup to ESC.\n'
    printf 'Training capture defaults to deferred rendering: HDF5 is generated automatically after ESC.\n'
    exit 0
fi
if [[ "$COMMAND" == doctor ]]; then
    for tool in docker python3 git curl xauth flock; do command -v "$tool"; done
    docker info --format 'Docker {{.ServerVersion}}; runtimes={{json .Runtimes}}'
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    printf 'DISPLAY=%s; session=%s\n' "${DISPLAY:-unset}" "${XDG_SESSION_TYPE:-unset}"
    printf 'Read README.md for hardware requirements and NVIDIA license acceptance.\n'
    exit 0
fi
if [[ "$COMMAND" == build ]]; then
    exec docker build -t "$IMAGE" .
fi
umask 0002
mkdir -p cache output .runtime/PhyRC_Sim .runtime/RCareWorld-2.0
exec 9>cache/runtime.lock
flock -n 9 || { printf 'This repository already has an active prepare/run. Close it first.\n' >&2; exit 2; }
for folder in kit ov pip warp glcache computecache logs data documents mpl; do
    mkdir -p "cache/$folder"
    chmod g+rwx "cache/$folder"
done
chmod g+rwx output .runtime .runtime/PhyRC_Sim .runtime/RCareWorld-2.0
PROJECT_COMMIT=$(git rev-parse HEAD)
PROJECT_DIRTY=0
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then PROJECT_DIRTY=1; fi
ARGS=(run --rm --network=host --user "1234:$(id -g)" --label phyrc.project=PhyRC_2027
      -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES
      -e "PHYRC_SOURCE_COMMIT=$PROJECT_COMMIT" -e "PHYRC_SOURCE_DIRTY=$PROJECT_DIRTY"
      -e MPLCONFIGDIR=/isaac-sim/.cache/matplotlib -e OPENBLAS_NUM_THREADS=1
      -v "$ROOT/.runtime/PhyRC_Sim:/workspace/PhyRC_Sim:rw"
      -v "$ROOT/.runtime/RCareWorld-2.0:/workspace/RCareWorld-2.0:rw"
      -v "$ROOT/output:/output:rw" -v "$ROOT/scripts:/scripts:ro"
      -v "$ROOT:/project:ro" -w /workspace/PhyRC_Sim)
for mapping in kit:kit/cache ov:.cache/ov pip:.cache/pip warp:.cache/warp glcache:.cache/nvidia/GLCache computecache:.nv/ComputeCache logs:.nvidia-omniverse/logs data:.local/share/ov/data documents:Documents mpl:.cache/matplotlib; do
    ARGS+=(-v "$ROOT/cache/${mapping%%:*}:/isaac-sim/${mapping#*:}:rw")
done
cpu() {
    docker "${ARGS[@]}" --entrypoint /bin/bash "$IMAGE" -c '
        umask 0002
        find /workspace/PhyRC_Sim -uid "$(id -u)" -exec chmod g+rwX {} +
        usd=(/isaac-sim/extscache/omni.usd.libs-*/)
        export PYTHONPATH="${usd[0]}:${PYTHONPATH:-}"
        export LD_LIBRARY_PATH="${usd[0]}/bin:/isaac-sim/kit/lib:${LD_LIBRARY_PATH:-}"
        exec /isaac-sim/python.sh "$@"
    ' bash "$@"
}
if [[ "$COMMAND" == prepare ]]; then
    python3 scripts/fetch_assets.py
    cp -a src/PhyRC_Sim/. .runtime/PhyRC_Sim/
    mkdir -p .runtime/RCareWorld-2.0/assets/robots
    cp -a assets/custom/robots/. .runtime/RCareWorld-2.0/assets/robots/
    mkdir -p .runtime/PhyRC_Sim/Assets/Human/Mesh/manikin_exports
    cp -a assets/custom/human/. .runtime/PhyRC_Sim/Assets/Human/Mesh/manikin_exports/
    mkdir -p .runtime/PhyRC_Sim/Assets/Material
    cp -a assets/downloads/Material/. .runtime/PhyRC_Sim/Assets/Material/
    install -D -m 664 assets/downloads/2K-tiling_30_basecolor.jpg \
        .runtime/PhyRC_Sim/Assets/Scene/kitchen/kitchen_7/textures/2K-tiling_30_basecolor.jpg
    find .runtime -uid "$(id -u)" -exec chmod g+rwX {} +
    cpu /scripts/prepare_assets.py
    exit 0
fi
if [[ "$COMMAND" == cpu ]]; then
    shift
    cpu "$@"
    exit 0
fi
[[ -f .runtime/PhyRC_Sim/Assets/Garment/Tops/Modelink/t_shirt_short.usd ]] || {
    printf 'Run ./run.sh prepare first.\n' >&2; exit 2;
}
ARGS+=(--gpus all)
# Only project-specific override names are forwarded, never arbitrary host secrets.
while IFS= read -r name; do
    case "$name" in STRETCH4_*|HUMAN_*|BOX_*|ROBOT_*|ROBOT1_*|ROBOT2_*|GARMENT_X_OFFSETS)
        ARGS+=(-e "$name");;
    esac
done < <(compgen -e)
if [[ "$COMMAND" == gui || "$COMMAND" == replay ]]; then
    : "${DISPLAY:?A local graphical desktop with an attached monitor is required}"
    command -v xauth >/dev/null
    touch cache/xauth
    chmod 600 cache/xauth
    cookie=$(xauth -f "${XAUTHORITY:-$HOME/.Xauthority}" nlist "$DISPLAY")
    [[ -n "$cookie" ]] || { printf 'No Xauthority cookie for DISPLAY=%s\n' "$DISPLAY" >&2; exit 2; }
    printf '%s\n' "$cookie" | sed 's/^..../ffff/' | xauth -f cache/xauth nmerge -
    chmod 644 cache/xauth
    ARGS+=(-e STRETCH4_HEADLESS=0 -e DISPLAY -e XAUTHORITY=/tmp/.docker.xauth
           -e "STRETCH4_SHOW_COLLIDER=${STRETCH4_SHOW_COLLIDER:-0}"
           -v "$ROOT/cache/xauth:/tmp/.docker.xauth:ro" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
    if [[ "$COMMAND" == replay ]]; then
        shift
        [[ $# -gt 0 ]] || { printf 'replay requires a recording directory\n' >&2; exit 2; }
        replay_path="$1"
        shift
        case "$replay_path" in
            "$ROOT"/output/*) replay_path="/output/${replay_path#"$ROOT"/output/}";;
            output/*) replay_path="/output/${replay_path#output/}";;
            ./output/*) replay_path="/output/${replay_path#./output/}";;
        esac
        exec docker "${ARGS[@]}" "$IMAGE" /scripts/replay_teleop.py "$replay_path" "$@"
    fi
    if [[ "${STRETCH4_TRAINING_RECORD:-0}" == 1 && "${STRETCH4_TRAINING_RENDER:-deferred}" == deferred ]]; then
        mkdir -p output/training_export_requests
        training_request="/output/training_export_requests/$(date -u +%Y%m%dT%H%M%S)_$$.json"
        if docker "${ARGS[@]}" -e "STRETCH4_TRAINING_EXPORT_REQUEST=$training_request" "$IMAGE" Env_StandAlone/Teleop_TShirt_Stretch4_Env.py; then
            if [[ -f "$ROOT${training_request}" ]]; then
                printf 'Teleop finished. Generating synchronized RGBD/HDF5 in a separate headless process...\n'
                exec docker "${ARGS[@]}" "$IMAGE" /scripts/export_policy_dataset.py --request "$training_request"
            fi
            printf 'No complete capture available for automatic export. Check the recording status.\n'
            exit 0
        else
            exit "$?"
        fi
    fi
    exec docker "${ARGS[@]}" "$IMAGE" Env_StandAlone/Teleop_TShirt_Stretch4_Env.py
elif [[ "$COMMAND" == smoke ]]; then
    exec docker "${ARGS[@]}" -e STRETCH4_HEADLESS=1 "$IMAGE" /scripts/smoke.py
elif [[ "$COMMAND" == python ]]; then
    shift
    exec docker "${ARGS[@]}" -e STRETCH4_HEADLESS=1 "$IMAGE" "$@"
fi
printf 'Unknown command: %s\n' "$COMMAND" >&2
exit 2
