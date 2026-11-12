#!/usr/bin/env bash
# Shared entry point; model identity and tuning come only from the reviewed locks.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
INSTALL_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --yes|--allow-driver-change) INSTALL_ARGS+=("$1"); shift ;;
    *) break ;;
  esac
done
command="${1:-start}"
if [ "$#" -gt 0 ]; then shift; fi
case "$command" in
  -h|--help|help)
    printf '%s\n' \
      'Usage: ./launch.sh [--yes] [--allow-driver-change] COMMAND [OPTIONS]' \
      'Default command: start (installs missing supported software, prepares model, starts server).' \
      'Commands: doctor install-deps setup print-command start status stop logs verify diagnostics reset-state' \
      'Routine installation --yes is separate from --accept-model-download.' \
      'Single-node start stays in the foreground; Ctrl-C stops only its owned process.'
    exit 0 ;;
  doctor|install-deps|setup|print-command|start|status|stop|logs|verify|diagnostics|reset-state) ;;
  *) printf 'Unknown command: %s\n' "$command" >&2; exit 2 ;;
esac
if [ "$command" = install-deps ] && [ "$#" -ne 0 ]; then
  printf '%s\n' 'Installer flags must precede install-deps; unexpected extra arguments.' >&2
  exit 2
fi
for arg in "$@"; do
  case "$arg" in
    -h|--help) bash "$ROOT/launch.sh" --help; exit 0 ;;
  esac
done
validate_start_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --profile|--site)
        if [ "$#" -lt 2 ] || [ -z "${2-}" ] || [[ "${2-}" = --* ]]; then
          printf '%s\n' 'A profile or site option needs a value.' >&2
          return 2
        fi
        shift 2 ;;
      --accept-model-download) shift ;;
      *) printf '%s\n' 'Unsupported startup option; use --profile, --site or --accept-model-download.' >&2; return 2 ;;
    esac
  done
}
case "$command" in
  start|setup) validate_start_args "$@" ;;
esac
if [ "$command" = start ] && [ "$EUID" -eq 0 ]; then
  printf '%s\n' 'RUN_AS_NORMAL_USER: launch without sudo; the installer requests sudo only for system packages.' >&2
  exit 2
fi
bootstrap() {
  # Bash 3.2 treats an empty array expansion as unset under nounset.
  if [ "${#INSTALL_ARGS[@]}" -eq 0 ]; then
    bash "$ROOT/scripts/bootstrap.sh"
  else
    bash "$ROOT/scripts/bootstrap.sh" "${INSTALL_ARGS[@]}"
  fi
}
case "$command" in
  install-deps) bootstrap; exit 0 ;;
  start|setup) bootstrap ;;
esac
command -v python3 >/dev/null 2>&1 || {
  printf '%s\n' 'DEPENDENCY_MISSING: run ./launch.sh install-deps to install supported prerequisites.' >&2
  exit 20
}
if [ "$command" = doctor ]; then
  exec python3 -I -B "$ROOT/scripts/doctor.py" "$@"
fi
exec python3 -I -B "$ROOT/scripts/runtime.py" "$command" "$@"
