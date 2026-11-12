#!/usr/bin/env bash
# Run only from an externally verified candidate archive. Before Python can
# validate the bundle, this shell may install ONLY the fixed Ubuntu interpreter
# closure below, never packages, repositories or hooks supplied by bundle JSON.
# Trusted base: Bash/coreutils, apt/dpkg, configured signed Ubuntu repositories,
# systemd 255/cgroup v2 (system manager and login user manager), and root or sudo.
# No repository/key provisioning, upgrades, drivers or reboot.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
yes=false
for arg in "$@"; do
  case "$arg" in
    --yes) yes=true ;;
    --allow-driver-change) ;;  # Forwarded only; never permits shell driver changes.
    *) printf '%s\n' 'INVALID_INSTALLER_OPTION: expected --yes or --allow-driver-change' >&2; exit 2 ;;
  esac
done
for file in profile.json model.lock.json release.lock.json dependencies.lock.json; do
  if [ ! -s "$ROOT/$file" ]; then
    printf '%s\n' 'PROFILE_NOT_READY: this is not a finalized runtime bundle.' >&2
    exit 2
  fi
done

fail() { printf '%s\n' "$2" >&2; exit "$1"; }
python_ready() {
  "$1" -I -B -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1
}
python=python3
if ! command -v python3 >/dev/null 2>&1 || ! python_ready python3; then
  # Fixed production paths, not caller overrides. Tests relocate these literals
  # in a disposable COPY and intercept every package command; no shipping bypass.
  OS_RELEASE='/etc/os-release'
  SYSTEMD_STATE='/run/systemd/system'
  CGROUP_ROOT='/sys/fs/cgroup'
  UUID_FILE='/proc/sys/kernel/random/uuid'
  PATH='/usr/sbin:/usr/bin:/sbin:/bin'
  python='/usr/bin/python3'
  export PATH LC_ALL=C LANG=C
  [ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ] ||
    fail 21 'UNSUPPORTED_INSTALLER_PLATFORM: bootstrap requires Ubuntu 24.04 ARM64'
  [ -r "$OS_RELEASE" ] || fail 21 'UNSUPPORTED_INSTALLER_PLATFORM: OS metadata unavailable'
  os_id= os_version= id_count=0 version_count=0
  # Parse only literal data; never source/eval os-release or bundle metadata.
  while IFS='=' read -r key value || [ -n "$key" ]; do
    case "$key" in
      ID) os_id=$value; id_count=$((id_count + 1)) ;;
      VERSION_ID) os_version=$value; version_count=$((version_count + 1)) ;;
    esac
  done < "$OS_RELEASE"
  case "$os_id" in ubuntu|'"ubuntu"'|"'ubuntu'") ;; *) fail 21 'UNSUPPORTED_INSTALLER_PLATFORM: expected Ubuntu' ;; esac
  case "$os_version" in 24.04|'"24.04"'|"'24.04'") ;; *) fail 21 'UNSUPPORTED_INSTALLER_PLATFORM: expected Ubuntu 24.04' ;; esac
  [ "$id_count" -eq 1 ] && [ "$version_count" -eq 1 ] ||
    fail 21 'UNSUPPORTED_INSTALLER_PLATFORM: ambiguous OS metadata'

  # A stale/custom PATH interpreter does not justify an OS package transaction.
  if ! python_ready "$python"; then
    for command in apt-get dpkg-query env timeout id systemd-run systemctl sleep; do
      command -v "$command" >/dev/null 2>&1 || fail 20 'BOOTSTRAP_BASE_REQUIRED: apt, dpkg, coreutils and systemd tools must already work'
    done
    approve() {
      printf '%s\n' "$1"
      if [ "$yes" = true ]; then return; fi
      [ -t 0 ] || fail 20 'APPROVAL_REQUIRED: rerun with --yes for routine installs'
      printf 'Continue? [y/N] '
      answer=
      read -r answer || true
      case "$answer" in y|Y|yes|YES|Yes) ;; *) fail 20 'INSTALL_DECLINED: no Python packages installed' ;; esac
    }
    approve 'Python 3.12 is missing/unusable. Refresh signed OS metadata and review a minimal python3 install?'
    privilege=()
    uid=$(id -u) || fail 20 'BOOTSTRAP_PRIVILEGE_FAILED: cannot determine OS user'
    if [ "$uid" != 0 ]; then
      command -v sudo >/dev/null 2>&1 || fail 20 'SUDO_REQUIRED: package installation requires root or sudo'
      # Only sudo itself reads a password from the terminal; never a pipe/log.
      if [ -t 0 ]; then
        sudo -v || fail 20 'BOOTSTRAP_PRIVILEGE_FAILED: sudo approval failed'
      else
        sudo -n -v || fail 20 'BOOTSTRAP_PRIVILEGE_FAILED: authorize sudo in a terminal first'
      fi
      privilege=(sudo -n)
    fi
    [ -d "$SYSTEMD_STATE" ] && [ -f "$CGROUP_ROOT/cgroup.controllers" ] && [ -r "$UUID_FILE" ] ||
      fail 20 'BOOTSTRAP_SUPERVISION_REQUIRED: Ubuntu 24.04 systemd and cgroup v2 must already work'
    control_env=(env -i "PATH=$PATH" LC_ALL=C LANG=C "XDG_RUNTIME_DIR=/run/user/$uid"
      "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$uid/bus")
    managers=(--system)
    [ "$uid" = 0 ] || managers+=(--user)
    for manager in "${managers[@]}"; do
      version=$("${control_env[@]}" timeout 5 systemctl "$manager" --no-ask-password show --property=Version --value 2>/dev/null) ||
        fail 20 'BOOTSTRAP_SUPERVISION_REQUIRED: systemd system/login user managers must be reachable'
      [[ "$version" =~ ^([0-9]+)([.[:space:]-]|$) ]] && [ "${BASH_REMATCH[1]}" -ge 255 ] ||
        fail 20 'BOOTSTRAP_SUPERVISION_REQUIRED: systemd 255 or newer is required'
    done

    bootstrap_unit_empty() {
      local mode=$1 unit=$2 child_rc=$3 status rc load= active= group= fields=0 key value populated=
      if status=$("${control_env[@]}" timeout 5 systemctl "$mode" --no-ask-password show \
          --all --property=LoadState,ActiveState,ControlGroup -- "$unit" 2>/dev/null); then rc=0; else rc=$?; fi
      [ "$rc" -le 1 ] || return 1
      while IFS='=' read -r key value; do
        case "$key" in
          LoadState) load=$value; fields=$((fields + 1)) ;;
          ActiveState) active=$value; fields=$((fields + 1)) ;;
          ControlGroup) group=$value; fields=$((fields + 1)) ;;
          *) return 1 ;;
        esac
      done <<< "$status"
      [ "$fields" -eq 3 ] || return 1
      if [ "$load" = not-found ]; then
        # Absence is ambiguous if the admission client was killed mid-request.
        [ "$child_rc" -lt 128 ] && [ "$active" = inactive ] && [ -z "$group" ]; return
      fi
      [ "$load" = loaded ] || return 1
      case "$active" in inactive|failed) ;; *) return 1 ;; esac
      [ -n "$group" ] || return 0
      case "$group" in /*/"$unit") ;; *) return 1 ;; esac
      case "$group/" in *'/../'*) return 1 ;; esac
      [ -e "$CGROUP_ROOT$group/cgroup.events" ] || return 0
      [ -r "$CGROUP_ROOT$group/cgroup.events" ] || return 1
      while read -r key value; do [ "$key" != populated ] || populated=$value; done < "$CGROUP_ROOT$group/cgroup.events"
      [ "$populated" = 0 ]
    }
    apt_owned() {
      local mode=$1 nonce unit pid before rc=0 signals=0 cancelled=0 warned=0 service_home=${HOME:-/}
      local elevation=()
      shift
      if [ "$mode" = --system ]; then
        elevation=(${privilege[@]+"${privilege[@]}"})
        service_home=/root
      fi
      IFS= read -r nonce < "$UUID_FILE" || return 20
      [[ "$nonce" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || return 20
      unit="rexl3-python-bootstrap-$nonce.service"
      # Only the system's env/apt are elevated, never candidate Python/scripts.
      # The timer is owned by the manager, not an unprivileged timeout wrapper.
      trap 'signals=$((signals + 1))' INT TERM HUP
      ${elevation[@]+"${elevation[@]}"} "${control_env[@]}" systemd-run "$mode" --no-ask-password \
        --quiet --wait --pipe --collect --service-type=exec --expand-environment=no --unit="$unit" \
        --property=ExitType=cgroup --property=KillMode=control-group --property=SendSIGKILL=yes \
        --property=Restart=no --property=TimeoutStartSec=30s --property=TimeoutStopSec=2s \
        --property=RuntimeMaxSec=900s -- env -i "PATH=$PATH" "HOME=$service_home" \
        LC_ALL=C LANG=C DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get "${apt_options[@]}" "$@" 2>/dev/null &
      pid=$!
      while :; do
        before=$signals
        if wait "$pid"; then rc=0; else rc=$?; fi
        [ "$signals" -ne "$before" ] || break
        cancelled=1
        # A denied stop is not a stopped transaction; keep waiting for the
        # manager-owned timer. Never kill sudo or scan/kill numeric descendants.
        ${elevation[@]+"${elevation[@]}"} "${control_env[@]}" timeout 5 systemctl "$mode" --no-ask-password stop -- "$unit" >/dev/null 2>&1 || :
      done
      while ! bootstrap_unit_empty "$mode" "$unit" "$rc"; do
        if [ "$warned" = 0 ]; then
          printf '%s\n' "BOOTSTRAP_DRAIN_PENDING: $unit; waiting for verified cgroup cleanup; ask the host administrator" >&2 || :
          warned=1
        fi
        ${elevation[@]+"${elevation[@]}"} "${control_env[@]}" timeout 5 systemctl "$mode" --no-ask-password stop -- "$unit" >/dev/null 2>&1 || :
        sleep .2 || :
      done
      trap - INT TERM HUP
      [ "$signals" -eq 0 ] && [ "$cancelled" -eq 0 ] || return 130
      return "$rc"
    }
    apt_options=(-o DPkg::Lock::Timeout=60 -o Acquire::Retries=0
      -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30
      -o APT::Update::Error-Mode=any -o APT::Get::AllowUnauthenticated=false
      -o Acquire::AllowInsecureRepositories=false
      -o Acquire::AllowDowngradeToInsecureRepositories=false)
    # env -i drops APT_CONFIG, credentials and loader/proxy overrides. Suppress
    # raw apt output (which may contain repository credentials); print fixed errors.
    apt_owned --system update </dev/null >/dev/null ||
      fail 20 'BOOTSTRAP_APT_UPDATE_FAILED: check network, signed repositories and package locks locally'
    simulation_manager=--user
    [ "$uid" != 0 ] || simulation_manager=--system
    plan=$(apt_owned "$simulation_manager" -s install --no-remove --no-upgrade --no-install-recommends python3 </dev/null) ||
      fail 20 'BOOTSTRAP_APT_PLAN_FAILED: minimal Python is unavailable from configured signed repositories'

    specs=() names=' ' planned_count= summary_count=0
    action_re='^(Inst|Conf) ([a-z0-9][a-z0-9+.-]*)(:arm64)? \(([0-9][A-Za-z0-9.+:~_-]*) [^()]*\[(arm64|all)\]\)$'
    summary_re='^0 upgraded, ([0-9]+) newly installed, 0 to remove and [0-9]+ not upgraded\.$'
    while IFS= read -r line; do
      case "$line" in
        *' upgraded,'*)
          [[ "$line" =~ $summary_re ]] || fail 20 'BOOTSTRAP_APT_PLAN_REFUSED: upgrades or removals are not allowed'
          planned_count=${BASH_REMATCH[1]}; summary_count=$((summary_count + 1)) ;;
        Remv\ *|Purg\ *) fail 20 'BOOTSTRAP_APT_PLAN_REFUSED: removals are not allowed' ;;
        Inst\ *|Conf\ *)
          # No [installed-version] field is permitted: even runtime-library
          # upgrades/reinstalls are refused, not just driver/kernel changes.
          [[ "$line" =~ $action_re ]] || fail 20 'BOOTSTRAP_APT_PLAN_REFUSED: unparseable or upgrading package action'
          action=${BASH_REMATCH[1]} name=${BASH_REMATCH[2]} version=${BASH_REMATCH[4]}
          case "$name" in
            python3|python3-minimal|python3.12|python3.12-minimal|libpython3-stdlib|libpython3.12-minimal|libpython3.12-stdlib|\
            libbz2-1.0|libcrypt1|libdb5.3t64|libexpat1|libffi8|liblzma5|libncursesw6|libreadline8t64|libsqlite3-0|\
            libssl3t64|libtinfo6|libuuid1|zlib1g|readline-common|media-types|netbase|tzdata) ;;
            *) fail 20 'BOOTSTRAP_APT_PLAN_REFUSED: package outside fixed Python runtime closure' ;;
          esac
          if [ "$action" = Inst ]; then
            case "$names" in *" $name "*) fail 20 'BOOTSTRAP_APT_PLAN_REFUSED: duplicate package action' ;; esac
            if status=$(env -i "PATH=$PATH" LC_ALL=C dpkg-query -W '-f=${Status}' "$name" 2>/dev/null); then
              case "$status" in
                'unknown ok not-installed'|'deinstall ok config-files'|'purge ok not-installed') ;;
                *) fail 20 'BOOTSTRAP_APT_PLAN_REFUSED: existing or partially installed package would change' ;;
              esac
            else
              [ "$?" -eq 1 ] || fail 20 'BOOTSTRAP_DPKG_FAILED: inspect the OS package database locally'
            fi
            specs+=("$name=$version")
            names="$names$name "
          else
            # Do not configure unrelated packages left pending by another install.
            case " ${specs[*]-} " in
              *" $name=$version "*) ;;
              *) fail 20 'BOOTSTRAP_APT_PLAN_REFUSED: unreviewed pending package configuration' ;;
            esac
          fi ;;
      esac
    done <<< "$plan"
    case "$names" in *' python3 '*) ;; *) fail 20 'BOOTSTRAP_APT_PLAN_REFUSED: no new python3 in the plan' ;; esac
    [ "$summary_count" -eq 1 ] && [ "$planned_count" -eq "${#specs[@]}" ] ||
      fail 20 'BOOTSTRAP_APT_PLAN_REFUSED: incomplete package plan'
    printf 'Reviewed minimal Python packages (new installs only):\n'
    printf '  %s\n' "${specs[@]}"
    approve 'Install this exact signed package plan? No upgrades, removals, drivers or kernel changes.'
    apt_owned --system install -y --no-remove --no-upgrade --no-install-recommends "${specs[@]}" </dev/null >/dev/null ||
      fail 20 'BOOTSTRAP_APT_INSTALL_FAILED: package installation failed; inspect the OS package manager locally'
    hash -r
    python_ready "$python" || fail 20 'BOOTSTRAP_PYTHON_READBACK_FAILED: installed Python 3.12 is unavailable or unusable'
  fi
fi
# System ensurepip is not an availability test (Ubuntu disables it). The existing
# isolated adapter validates ALL bundle data before handling venv/pip/dependencies.
if [ "$#" -eq 0 ]; then
  exec "$python" -I -B "$ROOT/scripts/install_dependencies.py"
fi
exec "$python" -I -B "$ROOT/scripts/install_dependencies.py" "$@"
