#!/usr/bin/env python3
"""Destructive controls for the DROID runtime process contract."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from qtail_droid_full_progress import (
    command_invokes_script,
    evaluate_runtime_process_contract,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def base_processes() -> dict[str, list[dict]]:
    pipeline_pid = 41001
    return {
        "pipeline": [{"pid": pipeline_pid, "ppid": 41000}],
        "watchdog": [{"pid": 41000, "ppid": 1}],
        "progress": [{"pid": 41002, "ppid": 1}],
        "prewarm": [{"pid": 41003, "ppid": 1}],
        "downloader": [{"pid": 41004, "ppid": pipeline_pid}],
        "handoff": [
            {
                "pid": 41005,
                "ppid": 1,
                "target_pipeline_pid": pipeline_pid,
            }
        ],
        "transport_guard": [{"pid": 41006, "ppid": 1}],
    }


def evaluate(processes: dict[str, list[dict]], age: float = 15.0) -> dict:
    return evaluate_runtime_process_contract(
        processes,
        stage="droid_full_download",
        heartbeat_age_seconds=age,
    )


def watchdog_handoff_source_valid(source: str) -> bool:
    marker_guard = 'if [ -f "$DOWNLOAD_MARKER" ]; then'
    exact_binding = (
        'local expected_handoff="/bin/zsh $HANDOFF $pipeline_pid"'
    )
    lookup = 'pgrep -f -x "$expected_handoff"'
    spawn = '/usr/bin/nohup /bin/zsh "$HANDOFF" "$pipeline_pid"'
    call = 'ensure_download_handoff "$pipeline_pid" || true'
    command_gate = (
        'if [ "$pipeline_command" != "$EXPECTED_COMMAND" ]; then'
    )
    required = [
        'HANDOFF="$ROOT/scripts/qtail_reload_pipeline_after_download.sh"',
        'DOWNLOAD_MARKER="$JOB_ROOT/manifests/DROID_DOWNLOAD_COMPLETE"',
        "ensure_download_handoff() {",
        marker_guard,
        exact_binding,
        lookup,
        spawn,
        call,
        command_gate,
    ]
    if not all(token in source for token in required):
        return False
    return bool(
        source.index(marker_guard) < source.index(spawn)
        and source.index(exact_binding) < source.index(lookup)
        and source.index(lookup) < source.index(spawn)
        and source.index(command_gate) < source.index(call)
    )


def launcher_source_valid(source: str) -> bool:
    mount_guard = (
        'if ! /sbin/mount | /usr/bin/grep -Fq '
        '" on /Volumes/ORICO ("; then'
    )
    web_services = (
        "/usr/bin/screen -dmS \"$WEB_SUPERVISOR_SESSION\" \\\n"
        "    /bin/zsh -lc \"exec /bin/zsh '$WEB_SERVICES'\""
    )
    write_probe = 'printf \'%s\\n\' "$$" > "$probe_path"'
    write_access_guard = "if ! probe_orico_write_access; then"
    write_access_refusal = (
        "ORICO write access unavailable; refusing to supervise or replace workers"
    )
    progress_call = (
        'qtail-droid-progress \\\n'
        '  "$PROGRESS_LOOP" \\\n'
        '  progress_healthy'
    )
    marker_guard = (
        'if [ ! -f "$MARKER_ROOT/DROID_DOWNLOAD_COMPLETE" ]; then'
    )
    prewarm_call = (
        'qtail-droid-prewarm \\\n'
        '    "$PREWARM_LOOP" \\\n'
        '    prewarm_healthy'
    )
    watchdog_call = (
        'qtail-droid-watchdog \\\n'
        '  "$WATCHDOG" \\\n'
        '  watchdog_healthy'
    )
    pipeline_lookup = 'pgrep -f -x "/bin/zsh $PIPELINE" 2>/dev/null'
    exact_binding = (
        'handoff_command="/bin/zsh $RELOAD_HANDOFF $pipeline_pid"'
    )
    handoff_lookup = 'pgrep -f -x "$handoff_command"'
    handoff_spawn = (
        "/bin/zsh -lc \"exec /bin/zsh '$RELOAD_HANDOFF' "
        "'$pipeline_pid'\""
    )
    required = [
        'MARKER_ROOT="/Volumes/ORICO/qtail_full_training/manifests"',
        'ORICO_WRITE_PROBE="$JOB_ROOT/.qtail-launcher-write-probe.$$"',
        "probe_orico_write_access() {",
        write_probe,
        write_access_guard,
        write_access_refusal,
        "file_fresh() {",
        'local target_path="$1"',
        "progress_healthy() {",
        "watchdog_healthy() {",
        "prewarm_heartbeat_valid() {",
        "prewarm_healthy() {",
        'file_fresh "$PREWARM_HEARTBEAT" 150 || return 1',
        'payload.get("control") != "droid_feature_prewarm_pid_heartbeat_v1"',
        'payload.get("status") != "alive"',
        'payload["pid"]',
        '[ "$heartbeat_pid" -eq "$expected_pid" ]',
        "stop_process_tree() {",
        'ensure_screen() {',
        'local health_check="$4"',
        'if [ "${#pids[@]}" -eq 1 ] && "$health_check"; then',
        "replacing stale/duplicate",
        'stop_process_tree "$pid"',
        mount_guard,
        web_services,
        progress_call,
        marker_guard,
        prewarm_call,
        watchdog_call,
        pipeline_lookup,
        exact_binding,
        handoff_lookup,
        handoff_spawn,
    ]
    if not all(token in source for token in required):
        return False
    return bool(
        source.index(mount_guard) < source.index(write_probe)
        and source.index(write_probe) < source.index(write_access_guard)
        and source.index(write_access_guard) < source.index(web_services)
        and source.index(web_services) < source.index(progress_call)
        and source.index(progress_call) < source.index(prewarm_call)
        and source.index(marker_guard) < source.index(prewarm_call)
        and source.index(prewarm_call) < source.index(watchdog_call)
        and source.rindex(marker_guard) < source.index(pipeline_lookup)
        and source.index(pipeline_lookup) < source.index(exact_binding)
        and source.index(exact_binding) < source.index(handoff_lookup)
        and source.index(handoff_lookup) < source.index(handoff_spawn)
    )


def web_services_source_valid(source: str) -> bool:
    archive_call = "archive_local_supervision_logs"
    foreign_gate = (
        'if port_is_listening "$port" && ! service_owned "$port"; then'
    )
    foreign_refusal = 'refusing to stop foreign listener on port $port'
    stop_owned_session = (
        '/usr/bin/screen -S "$session" -X quit >/dev/null 2>&1 || true'
    )
    start_owned_session = (
        '/usr/bin/screen -dmS "$session" /bin/zsh -lc'
    )
    required = [
        'PAGE_PATH="/qtail-droid-full-training"',
        'PAGE_MARKER="Q-Tail DROID Full Evidence"',
        "archive_local_supervision_logs() {",
        'temporary="$destination.$$.tmp"',
        '/bin/cp -p "$source" "$temporary"',
        '/bin/mv -f "$temporary" "$destination"',
        (
            "$ROOT/.tmp/qtail-droid-terminal-launcher.log|"
            "qtail_droid_terminal_launcher.log"
        ),
        (
            "$ROOT/.tmp/qtail-droid-launchd.err.log|"
            "qtail_droid_launchd_stderr.log"
        ),
        (
            "$ROOT/.tmp/qtail-droid-launchd.out.log|"
            "qtail_droid_launchd_stdout.log"
        ),
        (
            "$ROOT/.tmp/qtail-uniclash-guard.err.log|"
            "qtail_uniclash_guard_stderr.log"
        ),
        (
            "$ROOT/.tmp/qtail-uniclash-guard.out.log|"
            "qtail_uniclash_guard_stdout.log"
        ),
        (
            "$ROOT/.tmp/qtail-web-services.log|"
            "qtail_web_services_local.log"
        ),
        "service_owned() {",
        "content_healthy() {",
        "service_healthy() {",
        '"http://127.0.0.1:$port$PAGE_PATH"',
        '/usr/bin/grep -Fq "$PAGE_MARKER"',
        'if service_healthy "$port"; then',
        'expected="node $SERVE --symlinks -l tcp://0.0.0.0:$port"',
        "exec '$SERVE' --symlinks -l tcp://0.0.0.0:$port",
        foreign_gate,
        foreign_refusal,
        stop_owned_session,
        start_owned_session,
        "ensure_service 54655 qtail-web-54655",
        "ensure_service 6222 qtail-web-6222",
    ]
    if not all(token in source for token in required):
        return False
    return bool(
        source.rindex(archive_call)
        < source.index("ensure_service 54655 qtail-web-54655")
        and source.index(foreign_gate) < source.index(stop_owned_session)
        and source.index(foreign_refusal) < source.index(stop_owned_session)
        and source.index(stop_owned_session)
        < source.index(start_owned_session)
    )


def generation_handoff_source_valid(source: str) -> bool:
    watchdog_lookup = (
        'pgrep -f -x "$EXPECTED_WATCHDOG_COMMAND" 2>/dev/null'
    )
    watchdog_unique = (
        'if [ "${#watchdog_pids[@]}" -ne 1 ]; then'
    )
    pipeline_lookup = 'pgrep -f -x "$EXPECTED_COMMAND" 2>/dev/null'
    pipeline_unique = (
        'if [ "${#pipeline_pids[@]}" -ne 1 ] '
        '|| [ "${pipeline_pids[1]:-}" != "$TARGET_PID" ]; then'
    )
    marker_wait = "while true; do"
    marker_exists = 'if [ -f "$DOWNLOAD_MARKER" ]; then'
    marker_verifier = '"$PYTHON" "$DOWNLOAD_MARKER_VERIFIER"'
    marker_success = "then\n      break"
    command_gate = 'if [ "$command" != "$EXPECTED_COMMAND" ]; then'
    required = [
        watchdog_lookup,
        watchdog_unique,
        pipeline_lookup,
        pipeline_unique,
        marker_wait,
        marker_exists,
        marker_verifier,
        marker_success,
        command_gate,
    ]
    if not all(token in source for token in required):
        return False
    return bool(
        source.index(watchdog_lookup) < source.index(watchdog_unique)
        and source.index(watchdog_unique) < source.index(pipeline_lookup)
        and source.index(pipeline_lookup) < source.index(pipeline_unique)
        and source.index(pipeline_unique) < source.index(marker_wait)
        and source.index(marker_wait) < source.index(marker_exists)
        and source.index(marker_exists) < source.index(marker_verifier)
        and source.index(marker_verifier) < source.index(marker_success)
        and source.index(marker_wait) < source.index(command_gate)
    )


def generation_handoff_convergence_source_valid(
    *,
    handoff_source: str,
    watchdog_source: str,
    pipeline_source: str,
    prewarm_source: str,
) -> bool:
    watchdog_restart = (
        "/usr/bin/screen -dmS qtail-droid-watchdog \\\n"
        '  /bin/zsh -lc "exec /bin/zsh \'$WATCHDOG\'"'
    )
    handoff_prewarm_wait = (
        'while pgrep -f -x "$EXPECTED_PREWARM_COMMAND" '
        ">/dev/null 2>&1; do"
    )
    pipeline_restart = (
        'printf \'[%s] pipeline missing; restarting\\n\' '
        '"$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"'
    )
    pipeline_spawn = '/bin/zsh "$PIPELINE" >> "$LOG" 2>&1 &'
    pre_checksum_generation_gate = (
        'if ! require_pipeline_generation_marker "pre-checksum"; then'
    )
    checksum_commit = "if ! commit_checksum_marker; then"
    formal_prewarm_wait = (
        'while pgrep -f -x "/bin/zsh $PREWARM_LOOP" '
        ">/dev/null 2>&1; do"
    )
    prewarm_checksum_guard = (
        'if [ -f "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED" ]; then'
    )
    prewarm_exit_heartbeat = (
        'write_heartbeat "checksum_verified_exit" '
        '"$LAST_SHARD_COUNT" 0'
    )
    prewarm_exit = "exit 0"
    required_by_source = [
        (handoff_source, [watchdog_restart, handoff_prewarm_wait]),
        (watchdog_source, [pipeline_restart, pipeline_spawn]),
        (
            pipeline_source,
            [
                pre_checksum_generation_gate,
                checksum_commit,
                formal_prewarm_wait,
            ],
        ),
        (
            prewarm_source,
            [
                prewarm_checksum_guard,
                prewarm_exit_heartbeat,
                prewarm_exit,
            ],
        ),
    ]
    if not all(
        all(token in source for token in required)
        for source, required in required_by_source
    ):
        return False
    return bool(
        handoff_source.index(watchdog_restart)
        < handoff_source.index(handoff_prewarm_wait)
        and watchdog_source.index(pipeline_restart)
        < watchdog_source.index(pipeline_spawn)
        and pipeline_source.index(pre_checksum_generation_gate)
        < pipeline_source.rindex(checksum_commit)
        < pipeline_source.index(formal_prewarm_wait)
        and prewarm_source.index(prewarm_checksum_guard)
        < prewarm_source.index(prewarm_exit_heartbeat)
        < prewarm_source.index(prewarm_exit)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    positive = base_processes()
    missing_handoff = deepcopy(positive)
    missing_handoff["handoff"] = []
    duplicate_handoff = deepcopy(positive)
    duplicate_handoff["handoff"].append(
        {
            "pid": 41006,
            "ppid": 1,
            "target_pipeline_pid": 41001,
        }
    )
    stale_target = deepcopy(positive)
    stale_target["handoff"][0]["target_pipeline_pid"] = 40999
    duplicate_pipeline = deepcopy(positive)
    duplicate_pipeline["pipeline"].append({"pid": 41007, "ppid": 41000})
    missing_transport_guard = deepcopy(positive)
    missing_transport_guard["transport_guard"] = []
    post_download = {
        "pipeline": [{"pid": 41001, "ppid": 41000}],
        "watchdog": [{"pid": 41000, "ppid": 1}],
        "progress": [{"pid": 41002, "ppid": 1}],
        "prewarm": [],
        "downloader": [],
        "handoff": [],
        "transport_guard": [{"pid": 41006, "ppid": 1}],
    }
    watchdog_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "qtail_droid_pipeline_watchdog.sh"
    )
    watchdog_source = watchdog_path.read_text(encoding="utf-8")
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "qtail_droid_terminal_launcher.command"
    )
    launcher_source = launcher_path.read_text(encoding="utf-8")
    web_services_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "qtail_web_services.sh"
    )
    web_services_source = web_services_path.read_text(encoding="utf-8")
    generation_handoff_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "qtail_reload_pipeline_after_download.sh"
    )
    generation_handoff_source = generation_handoff_path.read_text(
        encoding="utf-8"
    )
    pipeline_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "qtail_orico_full_pipeline.sh"
    )
    pipeline_source = pipeline_path.read_text(encoding="utf-8")
    prewarm_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "qtail_droid_feature_prewarm_loop.sh"
    )
    prewarm_source = prewarm_path.read_text(encoding="utf-8")
    missing_marker_guard = watchdog_source.replace(
        'if [ -f "$DOWNLOAD_MARKER" ]; then',
        'if false; then',
        1,
    )
    unbound_handoff = watchdog_source.replace(
        'local expected_handoff="/bin/zsh $HANDOFF $pipeline_pid"',
        'local expected_handoff="/bin/zsh $HANDOFF"',
        1,
    )
    launcher_missing_mount_guard = launcher_source.replace(
        'if ! /sbin/mount | /usr/bin/grep -Fq '
        '" on /Volumes/ORICO ("; then',
        "if false; then",
        1,
    )
    launcher_without_write_access_guard = launcher_source.replace(
        "if ! probe_orico_write_access; then",
        "if false; then",
        1,
    )
    launcher_missing_resume_call = launcher_source.replace(
        "watchdog_healthy() {",
        ": # watchdog resume call removed",
        1,
    )
    launcher_missing_marker_guard = launcher_source.replace(
        'if [ ! -f "$MARKER_ROOT/DROID_DOWNLOAD_COMPLETE" ]; then',
        "if true; then",
        1,
    )
    launcher_unbound_handoff = launcher_source.replace(
        'handoff_command="/bin/zsh $RELOAD_HANDOFF $pipeline_pid"',
        'handoff_command="/bin/zsh $RELOAD_HANDOFF"',
        1,
    )
    launcher_without_freshness = launcher_source.replace(
        'if [ "${#pids[@]}" -eq 1 ] && "$health_check"; then',
        'if [ "${#pids[@]}" -eq 1 ]; then',
        1,
    )
    launcher_without_tree_cleanup = launcher_source.replace(
        'stop_process_tree "$pid"',
        ": # process-tree cleanup removed",
        1,
    )
    launcher_without_prewarm_pid_binding = launcher_source.replace(
        '[ "$heartbeat_pid" -eq "$expected_pid" ]',
        '[ "$heartbeat_pid" -gt 0 ]',
        1,
    )
    launcher_with_zsh_path_shadow = launcher_source.replace(
        'local target_path="$1"',
        'local path="$1"',
        1,
    )
    web_services_without_content_marker = web_services_source.replace(
        '/usr/bin/grep -Fq "$PAGE_MARKER"',
        "/usr/bin/grep -Fq '<title>unrelated page</title>'",
        1,
    )
    web_services_without_foreign_listener_gate = web_services_source.replace(
        'if port_is_listening "$port" && ! service_owned "$port"; then',
        "if false; then",
        1,
    )
    web_services_without_symlink_resolution = web_services_source.replace(
        " --symlinks",
        "",
    )
    handoff_without_unique_watchdog = generation_handoff_source.replace(
        'if [ "${#watchdog_pids[@]}" -ne 1 ]; then',
        "if false; then",
        1,
    )
    handoff_without_unique_pipeline = generation_handoff_source.replace(
        'if [ "${#pipeline_pids[@]}" -ne 1 ] '
        '|| [ "${pipeline_pids[1]:-}" != "$TARGET_PID" ]; then',
        "if false; then",
        1,
    )
    handoff_without_marker_binding = generation_handoff_source.replace(
        '"$PYTHON" "$DOWNLOAD_MARKER_VERIFIER"',
        ": # semantic download-marker verifier removed",
        1,
    )
    handoff_without_early_watchdog = generation_handoff_source.replace(
        "/usr/bin/screen -dmS qtail-droid-watchdog \\\n"
        '  /bin/zsh -lc "exec /bin/zsh \'$WATCHDOG\'"',
        ": # current-generation watchdog start removed",
        1,
    )
    watchdog_without_pipeline_restart = watchdog_source.replace(
        '/bin/zsh "$PIPELINE" >> "$LOG" 2>&1 &',
        ": # current-generation pipeline restart removed",
        1,
    )
    pipeline_without_pretraining_checksum_commit = pipeline_source.replace(
        "if ! commit_checksum_marker; then",
        "if false; then",
        pipeline_source.count("if ! commit_checksum_marker; then"),
    )
    prewarm_without_checksum_exit = prewarm_source.replace(
        'if [ -f "$MARKER_ROOT/DROID_CHECKSUM_VERIFIED" ]; then',
        "if false; then",
        1,
    )

    guard_script = Path(
        "/Users/avalok/work/Q-TAIL-MVP/tools/"
        "qtail_uniclash_transport_guard.py"
    )
    process_classifier_control = bool(
        command_invokes_script(
            (
                "/Library/Frameworks/Python.framework/Versions/3.12/"
                "Resources/Python.app/Contents/MacOS/Python "
                f"{guard_script} --status /tmp/guard.json"
            ),
            guard_script,
        )
        and not command_invokes_script(
            f"/bin/zsh -c ps -axo command= | rg '{guard_script}'",
            guard_script,
        )
    )

    checks = {
        "positive_download_contract_passes": (
            evaluate(positive)["passed"]
            and process_classifier_control
        ),
        "missing_handoff_is_rejected": not evaluate(missing_handoff)[
            "passed"
        ],
        "duplicate_handoff_is_rejected": not evaluate(duplicate_handoff)[
            "passed"
        ],
        "stale_handoff_target_is_rejected": not evaluate(stale_target)[
            "passed"
        ],
        "duplicate_pipeline_is_rejected": not evaluate(duplicate_pipeline)[
            "passed"
        ],
        "stale_download_heartbeat_is_rejected": not evaluate(
            positive, age=301.0
        )["passed"],
        "missing_transport_guard_is_rejected": not evaluate(
            missing_transport_guard
        )["passed"],
        "post_download_contract_drops_transfer_processes": (
            evaluate_runtime_process_contract(
                post_download,
                stage="checksum_verification",
                heartbeat_age_seconds=None,
            )["passed"]
        ),
        "launcher_and_watchdog_self_healing_contract_passes": (
            watchdog_handoff_source_valid(watchdog_source)
            and launcher_source_valid(launcher_source)
            and web_services_source_valid(web_services_source)
        ),
        "launcher_without_prewarm_pid_binding_is_rejected": (
            not launcher_source_valid(launcher_without_prewarm_pid_binding)
        ),
        "source_without_resume_or_mount_contract_is_rejected": (
            not watchdog_handoff_source_valid(missing_marker_guard)
            and not launcher_source_valid(launcher_missing_mount_guard)
            and not launcher_source_valid(
                launcher_without_write_access_guard
            )
            and not launcher_source_valid(launcher_missing_resume_call)
            and not launcher_source_valid(launcher_missing_marker_guard)
            and not launcher_source_valid(launcher_without_freshness)
            and not launcher_source_valid(launcher_without_tree_cleanup)
            and not launcher_source_valid(launcher_with_zsh_path_shadow)
            and not web_services_source_valid(
                web_services_without_content_marker
            )
            and not web_services_source_valid(
                web_services_without_foreign_listener_gate
            )
            and not web_services_source_valid(
                web_services_without_symlink_resolution
            )
        ),
        "source_without_pipeline_pid_binding_is_rejected": (
            not watchdog_handoff_source_valid(unbound_handoff)
            and not launcher_source_valid(launcher_unbound_handoff)
        ),
        "generation_handoff_exact_ownership_contract_passes": (
            generation_handoff_source_valid(generation_handoff_source)
        ),
        "generation_handoff_without_unique_owners_or_marker_binding_is_rejected": (
            not generation_handoff_source_valid(
                handoff_without_unique_watchdog
            )
            and not generation_handoff_source_valid(
                handoff_without_unique_pipeline
            )
            and not generation_handoff_source_valid(
                handoff_without_marker_binding
            )
        ),
        "generation_handoff_checksum_convergence_contract_passes": (
            generation_handoff_convergence_source_valid(
                handoff_source=generation_handoff_source,
                watchdog_source=watchdog_source,
                pipeline_source=pipeline_source,
                prewarm_source=prewarm_source,
            )
        ),
        "generation_handoff_missing_convergence_edge_is_rejected": (
            not generation_handoff_convergence_source_valid(
                handoff_source=handoff_without_early_watchdog,
                watchdog_source=watchdog_source,
                pipeline_source=pipeline_source,
                prewarm_source=prewarm_source,
            )
            and not generation_handoff_convergence_source_valid(
                handoff_source=generation_handoff_source,
                watchdog_source=watchdog_without_pipeline_restart,
                pipeline_source=pipeline_source,
                prewarm_source=prewarm_source,
            )
            and not generation_handoff_convergence_source_valid(
                handoff_source=generation_handoff_source,
                watchdog_source=watchdog_source,
                pipeline_source=pipeline_without_pretraining_checksum_commit,
                prewarm_source=prewarm_source,
            )
            and not generation_handoff_convergence_source_valid(
                handoff_source=generation_handoff_source,
                watchdog_source=watchdog_source,
                pipeline_source=pipeline_source,
                prewarm_source=prewarm_without_checksum_exit,
            )
        ),
    }
    payload = {
        "generated_at": now(),
        "status": "passed" if all(checks.values()) else "failed",
        "control": "droid_runtime_process_contract_v11",
        "checks": checks,
        "checks_passed": sum(value is True for value in checks.values()),
        "checks_total": len(checks),
        "claim_boundary": (
            "This proves process-count, heartbeat, handoff-target, and "
            "launcher mount/write-permission/resume, PID-bound prewarm "
            "liveness, watchdog "
            "self-healing, exact handoff "
            "binding, stale-loop heartbeat replacement, recursive descendant "
            "cleanup, exact Python-script process classification, content-"
            "checked owned web-service recovery, audited symlink-artifact "
            "resolution, foreign-listener refusal, "
            "atomic ORICO supervision-log archival, and "
            "detached-screen-safe unique ownership plus semantic download-"
            "marker source-gate behavior. It also proves the source-order "
            "convergence path from post-download watchdog reload through "
            "checksum-marker commit to natural prewarm exit on "
            "controlled inputs. The launcher source gate also proves that a "
            "macOS privacy domain that can see but cannot write ORICO exits "
            "before stopping or replacing any worker. "
            "Live process identity is separately checked by "
            "qtail_droid_full_progress.py."
        ),
    }
    write_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
