# Q-Tail DROID Full Allocation Training Protocol

## Claim boundary

This run evaluates whether a Q-Tail allocation head can assign more fixed
probability mass to rare or risky DROID shards than a source-distribution head
on held-out data. It does not measure end-to-end robot-policy success.

## Immutable inputs

- Official source: `gs://gresearch/robotics/droid`
- Expected objects: 4,102
- Expected bytes: 3,700,745,265,151
- TFRecord shards: 4,096
- Official episodes: 187,891
- Releases:
  - `1.0.0/r2d2_faceblur`: 2,048 shards, 92,233 episodes
  - `1.0.1/droid_101`: 2,048 shards, 95,658 episodes
- Empirical PT source: `data/uploaded_data.csv`
- PT source SHA-256:
  `59e487af80482215b2c2d4e81e9ccd7471ac6c94c1ef40547596ccb80367e75f`

Formal training remains closed until every official object passes the local
MD5 ledger and the final `gsutil rsync -c` mirror check.

`DROID_SOURCE_PROBED` is a structured, atomic marker with format
`qtail_droid_source_probe_marker_v2`. It binds the official source URI, exact
remote byte total, ORICO job root, capacity decision, source-probe report path,
and report SHA-256. A valid retained report can be resealed with
`qtail_probe_droid_source.py --seal-existing` without making another network
request; the semantic marker verifier still checks all report bindings. This
proves the retained source-probe evidence has not drifted. It does not replace
the live capacity gate or independently re-query the bucket.

The workspace keeps only external symbolic links for large or authoritative
assets: `data/openx_demo`, `data/droid`, the official
`droid_policy_learning` checkout, and `results/qtail_droid_full` all resolve
into `/Volumes/ORICO/qtail_full_training`. The migration completion gate
resolves and verifies each link rather than trusting its displayed path.

The reproducibility-code snapshot is published by
`tools/qtail_publish_orchestration_snapshot.py`. It first rebuilds and
source-verifies every manifest path in a sibling directory on ORICO, holds the
same `.progress_refresh.lock` used by the public status builder, and then uses
macOS `renameatx_np(RENAME_SWAP)` to exchange the complete directories in one
same-volume atomic operation. While retaining that lock, it reverifies the
published tree and commits the publication audit before allowing any status
refresh. This prevents the live completion audit from
observing a new source file with an old `SHA256SUMS`, or the inverse. The
publisher is included in its own next-generation snapshot and writes
`qtail_orchestration_snapshot_sync_audit.json`; the page QA requires that
audit and the published `SHA256SUMS` to remain directly accessible. Completion
requirement 1 also rejects the snapshot unless the audit path, manifest hash,
file counts, progress-lock assertion, and atomic-swap method all match the live
verified tree. Malformed numeric fields close the gate rather than crashing
the status loop. This is a local consistency and reproducibility guarantee,
not an external timestamp or WORM claim.

ORICO capacity is checked continuously with the
`official_md5_plus_allocated_resumable_parts_v2` model. Completed objects count
as reusable only while their official-MD5 ledger entry still matches path,
size, timestamps, generation, and local bytes. Incomplete objects count only
allocated filesystem blocks in the exact manifest-bound `.qtail.part`; inflight,
invalid, quarantined, or unrelated files do not reduce the required capacity.
The gate requires current free bytes to cover all remaining official bytes plus
a fixed 5% reserve (185,037,263,258 bytes). This is a storage-allocation claim,
not an integrity claim: every resumed object must still pass final official MD5.
The downloader itself defaults to the same fixed reserve and the pipeline also
passes it explicitly, so the next naturally launched downloader generation
enforces the limit even if its parent shell was started from an older pipeline
revision. The currently running generation is not restarted merely to activate
this additional guard, preserving download continuity; the live status reports
the process-generation activation boundary.
The downloader guard self-test contains thirteen positive and destructive
controls: five single-writer lifecycle checks plus exact-floor, one-byte-short,
manifest-bound partial, oversize-part, pre-payload reserve rejection, and
policy-recording checks, plus presence and removal controls for the physical
interface binding. Final completion requires all thirteen to pass.

The production downloader is pinned to `direct` transport. Direct mode adds
`curl --noproxy '*' --interface en1`, so desktop proxy settings and inherited
proxy environment variables cannot silently route multi-TiB payloads through a
metered VPN, while curl itself is unable to migrate away from the physical
Wi-Fi interface. The first two Google Storage endpoints are preferred for
direct downloads; endpoint rotation remains available for retry recovery. The final
`gsutil rsync -c` verification also clears proxy environment variables and
sets `NO_PROXY=*`.

The live downloader uses HTTP/2 with 16 independent workers. This setting was
chosen from direct, bounded Range-request measurements against official DROID
objects while the production download remained active: 8 workers delivered
2.43 MiB/s, 16 delivered 2.80 MiB/s, and 24 regressed to 1.26 MiB/s because
of connection-tail latency. A same-object 8 MiB control measured HTTP/2 at
621 KB/s versus HTTP/1.1 at 461 KB/s. These are transport-tuning observations,
not model-quality evidence; the status file records the active protocol,
worker count, endpoint policy version, and exact route observations.
After the production handoff on 2026-07-28, the first 180-second window
delivered 1.726 MiB/s versus 1.009 MiB/s in the preceding 120-second
8-worker/HTTP/1.1 window, a 71.07% transport-throughput increase. Both windows
ran with UniClashCore online and zero forbidden or wrong-route observations.
The bounded controls, production windows, selected configuration, raw-log
SHA-256, code hashes, route observations, and explicit non-model claim boundary
are sealed in `droid_transport_tuning_audit.json`.

Formal downloads also enable `--forbid-tunnel-route`. Before every 64 MiB
Range request, the downloader resolves the active Google endpoint and inspects
the macOS route. The route must equal `en1`; interfaces matching `utun`, `tun`,
`tap`, `ppp`, `ipsec`, `gif`, `stf`, or `lo`, plus loopback gateways, are
rejected before `curl` starts. Each curl is independently bound with
`--interface en1`. Accepted interface, gateway, endpoint IP, and binding policy
are written to `parallel_download_status.json`.

UniClashCore is required to remain online during the download, but DROID
payload traffic must bypass it. A launchd-backed guard checks this contract
every two seconds. It requires Core running, TUN disabled, the Google Storage
bypass domains present, every active DROID `curl` carrying both
`--noproxy '*'` and `--interface en1`, no explicit proxy argument or
loopback/proxy-port socket, and every observed Google route using `en1`. A
violating DROID transfer is terminated while the
pipeline remains resumable. Current observations and any blocked process are
written to `uniclash_transport_guard.json`; this guard is one of the nine
completion requirements.

The live guard file continues changing after training because UniClashCore is
still monitored. Final page QA therefore writes
`uniclash_transport_guard_final.json` as an immutable completion-time snapshot.
The live file remains the operational status source; only the immutable
snapshot enters the final artifact SHA-256 manifest.
The same rule applies to the minute-level download history:
`download_progress_samples.json` remains live, while
`download_progress_samples_final.json` freezes the complete history for the
final artifact manifest.
The page renders at most 240 points from the retained hash-chained pipeline
timeline. Uniform index sampling is augmented with both sides of every
kind/stage transition, and always preserves the first and latest sample.
Missing historical MD5, preparse, or record fields remain null and are not
interpolated. The bounded chart is a rendering view only; the complete timeline
and its SHA-256 chain remain authoritative.
During download, `qtail_verify_droid_page.mjs --smoke` renders both
1,440-by-1,000 desktop and 390-by-844 mobile views without creating a final QA
marker. Its `live_page_smoke.json` output records overflow, canvas, response,
console, claim-boundary, intermediate-evidence, and 4/9 non-completion checks.
This is ongoing UI evidence only; final 9/9 QA is rerun after formal training.
The minute-level progress loop also runs `qtail_web_services.sh`. Each endpoint
must have the expected `serve --symlinks` command and the DROID page content
marker, not merely an open port. Symlink resolution is required because the
served result tree intentionally exposes immutable marker files from the
same ORICO job root; browser QA still requires every READY artifact URL to
return HTTP 200. An unhealthy project-owned screen is restarted, while a
foreign listener is logged and never terminated. Recovery events are retained
in `/Volumes/ORICO/qtail_full_training/logs/qtail-web-services.log`.

During guard development, a controlled gsutil probe exposed a classifier-v1
false positive: its launcher shell contained the gsutil token and DROID URI
but had no forbidden socket or wrong route. The raw status is preserved as
`uniclash_transport_guard_v1_classifier_false_positive.json`, with the
reasoning and SHA-256 linkage in
`uniclash_transport_guard_adjudication.json`. Classifier v2 limited matching
to the actual gsutil executable (or Python-launched `gsutil.py`) and
descendants. A second controlled probe then exposed environment elision for
some descendants; that raw epoch is also preserved. Classifier v3 validates
the clean proxy environment on the gsutil root process and applies that result
to descendants, while still checking every descendant socket route
independently. This avoids treating macOS `ps` environment elision on worker
processes as a transfer violation. When the downloader added the JSON media
endpoint, v3 then exposed a coverage gap: literal `robotics/droid` URLs were
matched, but the equivalent `robotics%2Fdroid` form was undercounted. The raw
v3 epoch is preserved as
`uniclash_transport_guard_v3_encoded_path_underobservation.json`.

Classifier v4 decodes official URLs and also walks the
`qtail_parallel_gcs_download.py` process tree, so every descendant curl is
covered independently of endpoint spelling. On 2026-07-30, UniClashCore briefly
restarted; the fail-closed guard terminated active transfers even though every
observed socket still routed through `en1`, with zero loopback/7993 and zero
wrong-route observations. That raw v4 state is preserved rather than erased.
Classifier v5 then rejected the still-running pre-binding curl generation
during migration. Its raw state also records zero forbidden sockets and zero
wrong routes. Classifier v6 begins only after the resumable downloader was
relaunched with `--interface en1` on every worker. Its positive/negative
controls are sealed in
`uniclash_transport_guard_classifier_v6_selftest.json`.
`uniclash_transport_guard_adjudication.json` binds all five prior archives,
the v6 self-test, and live bound-interface/socket/route evidence by SHA-256.
Completion requires the clean v6 epoch and every archived adjudication hash to
match.

Immediately before every final `gsutil rsync -r -c` attempt, the pipeline runs
an independent launch gate against the live guard heartbeat. The gate requires
UniClashCore to remain online, TUN to remain disabled, Google Storage system
proxy bypass entries to be present, both `curl` and `gsutil` to be covered by
the guard, a transport-clean cumulative history, and every currently observed
route to use `en1`. Gate v2 retains and accepts a historical Core-off heartbeat
only when that event contains no DROID process or transfer violation and the
entire epoch still has zero blocked PIDs, forbidden proxy sockets, and wrong
routes. This represents an idle machine/reboot policy pause, not an inferred
clean transfer; any Core-off event during a transfer remains a hard failure.
A stale or failed gate writes
`uniclash_pre_checksum_gate.json`, withholds `gsutil`, and retries after 30
seconds without stopping UniClashCore or the resumable pipeline. Thirteen synthetic
positive/destructive controls are sealed in
`uniclash_pre_checksum_gate_selftest.json`. This launch assertion complements,
and never replaces, continuous socket/route observation while `gsutil` runs.
Runtime health also requires exactly one transport-guard process. Each
hash-chained timeline sample records the guard file timestamp, its age, and the
live guard-process count; after this freshness contract is activated, an age
above ten seconds, a missing guard, or a duplicate guard invalidates the
timeline and completion gate. Earlier retained timeline samples are preserved
under an explicit pre-activation boundary rather than retroactively claiming
freshness fields that were not collected.
The checksum command itself runs as a monitored child process. While it is
alive, the same gate is re-evaluated every two seconds with a six-second
freshness limit. If the guard heartbeat becomes stale or any isolation check
fails, the pipeline terminates the complete `gsutil` process tree, records
return code 86, and retries only after the launch gate becomes healthy again.
The same direct-route transition gate is re-run before checksum-marker
commit, environment capture, and formal training launch. Thus UniClashCore
stays online across the whole pipeline while every network-capable DROID
handoff remains bound to the clean `en1` route contract.

## Mirror closure

Before the expensive checksum closure starts, the pipeline independently parses
the official `dataset_info.json` and `features.json` files for releases 1.0.0
and 1.0.1. `droid_release_metadata_audit.json` binds all four files to the
official object checksum manifest and requires the exact release identities,
2,048 shards and 92,233 records for `r2d2_faceblur`, 2,048 shards and 95,658
records for `droid_101`, 4,096 shards and 187,891 records combined, and an
identical required step schema. This is a metadata/schema gate only; it never
substitutes for downloading, hashing, and decoding every shard and record.

After the checksum sync succeeds, the pipeline requires all 4,102 official
targets to exist at their manifest sizes before it touches transport
artifacts. Recognized `.qtail.part*` and `.invalid-*` files are moved to a
results-side quarantine and recorded in
`droid_transport_cleanup_audit.json`. Unknown extra files are never removed
automatically; they fail the cleanup preflight and keep training closed.

The final mirror verifier then requires exactly 4,102 data files, zero missing
objects, zero size mismatches, zero extra or partial files, and exactly
3,700,745,265,151 official bytes. It also binds every local file's current
size, APFS `mtime`/`ctime`, official MD5, locally observed MD5, and generation
to the 4,102-entry checksum ledger. A structured checksum marker is accepted
only after this verifier passes. On restart, the fast path first verifies that
the marker still hashes the prior successful byte-checksum report (official
`gsutil rsync -c`, explicit local MD5 rehash, or both), then checks
all 4,102 current file sizes and `mtime`/`ctime` values against the bound ledger
in `droid_checksum_stat_continuity.json`. Any marker, report, path, size, or
timestamp mismatch invalidates the fast path and falls back to a complete local
MD5 rehash; unchanged data therefore keeps its prior byte-level proof without
re-reading 3.7 TB solely because orchestration code changed.

`DROID_CHECKSUM_VERIFIED` is written atomically only after it binds both the
immutable download-completion marker and `download_verification.json` by path,
byte count, and SHA-256. Marker-write failure is fatal. The pipeline records
structured terminal events for checksum closure, 187,891-record closure,
training completion, and final QA commit; the final process-log verifier
requires those events rather than accepting hashes of merely present logs.

If the final `gsutil rsync -c` repairs an object, the resulting APFS identity
can legitimately make an earlier ledger entry stale. A failed post-rsync
binding therefore atomically invalidates `DROID_DOWNLOAD_COMPLETE`. The
watchdog restarts the resumable downloader, which re-hashes complete targets,
refreshes their official MD5 ledger entries, and returns to mirror closure.
This prevents an otherwise-correct repaired mirror from entering a permanent
restart loop while never weakening the checksum gate.

The final environment manifest independently re-parses the object manifest,
checksum manifest, and `download_verification.json`. It requires unique paths,
matching object counts and byte totals, an MD5 entry for every object, checksum
rsync return code zero, a 1.0 local-to-remote ratio, and zero missing,
mismatched, extra, or partial files. Mere file existence cannot open the
training gate. A bounded positive control passes this contract; changing the
verified local byte total by one byte is a required negative-control failure.

The hash-chained `pipeline_timeline.json` retains every progress sample.
`pipeline_timeline_current_verification.json` is regenerated each minute and
checks every retained full-pipeline sample, not only the latest snapshot, for
Core online, TUN disabled, and zero blocked, forbidden-socket, or wrong-route
observations. Legacy byte-only samples predate these transport fields and are
explicitly excluded from that route claim. Final QA freezes and independently
re-verifies the complete chain.

## Record evidence

Every decodable episode in every complete TFRecord shard is streamed. One
audited row is cached per shard using its exact size, APFS `mtime`/`ctime`,
first/last 1 MiB boundary SHA-256, record count, and feature-extractor version.
The background prewarm process is pinned to the same Python 3.12 interpreter as
formal training and remains active until `DROID_CHECKSUM_VERIFIED`, not merely
until the first download-complete signal. The generation handoff waits for any
old atomic cache pass to finish before starting the current prewarm code, so it
never creates concurrent cache writers.

Every successful prewarm pass also writes
`droid_incremental_closure_audit.json`. It joins the official checksum
manifest, local MD5 ledger, live file identity, exact listed feature caches,
and official `shardLengths` audit. Transport partials and unreferenced cache
files are explicitly excluded. Each cache manifest binds the source snapshot
timestamp, shard count, and sorted shard-path SHA-256. A TFRecord whose MD5
verification completes after that snapshot is reported as
`deferred_after_snapshot` and is not counted as cached until the next prewarm
pass; a pre-snapshot TFRecord without a listed cache remains a hard failure.
The prewarm status distinguishes that snapshot scope explicitly. A successful
pass below 4,096 official TFRecords is
`prewarm_caught_up_current_snapshot`: every TFRecord complete at the pass
snapshot was decoded and cached, but full DROID coverage is not claimed. Only
an exact 4,096-shard snapshot with successful parse and scan coverage and both
official releases at exactly 2,048 shards may use
`prewarm_full_official_shard_snapshot_complete`; checksum and record closure
remain separate mandatory formal-training gates.
The six-case status-scope control artifact is rerun by the post-download
pipeline. Final browser QA requires the exact six control names, `6/6`, and
all-true results, then includes the artifact in the final SHA-256 manifest.
Its status remains `passed_incremental` until all 4,102 objects, 4,096
TFRecords, and 187,891 records close with zero deferred or missing shards; only
then may `formal_full_mirror_gate` become true.
Per-release mutable observations are written to
`release_milestones/droid_release_<version>_progress.json`. The corresponding
`*_complete.json` path does not exist while a release is waiting; it is created
only after exact object/MD5, 2,048-shard, byte, official-record, and
zero-partial closure, and is application-level append-once thereafter. The
sealer compares stable fields on every later run and fails on mismatch instead
of replacing the completed payload. This is not an APFS immutable flag,
external timestamp, or WORM guarantee. Legacy waiting placeholders formerly
stored under the complete filename are atomically migrated to the progress
filename without discarding their evidence.
`droid_incremental_closure_selftest.json` runs seven exact controls: the
positive current closure, an exact-full formal-mode rejection, destructive
record-count, MD5-ledger, post-200-error MD5-ledger, and missing-cache
controls, plus a controlled post-snapshot TFRecord. The formal rejection and
post-snapshot controls share a frozen fixture in which exactly one otherwise
valid TFRecord is deferred after the snapshot: ordinary incremental closure
must pass, while `--require-formal` must reject it even when the live mirror is
already complete. All four destructive mutations must be rejected, while the
post-snapshot object must be explicitly deferred without opening the formal
gate; the post-200 case proves bounded error display cannot hide a later
integrity failure. Because the downloader
can update its ledger while these controls run, the self-test first freezes one
byte-identical snapshot of all four mutable input documents. The controlled
deferral must increase that frozen snapshot's observed deferral count by
exactly one; it is not compared with a hard-coded zero-activity baseline.

The transition out of the download stage is also bound rather than represented
by an unstructured sentinel. `DROID_DOWNLOAD_COMPLETE` is an atomically written
`droid_download_completion_marker_v1` document that binds the 4,102-object
manifest, official MD5 manifest, exact ledger identity, 4,096 TFRecords,
3,700,745,265,151 bytes, completed transport status, and direct-route guard.
The pipeline verifies this binding on every start. A legacy zero-byte sentinel
may be upgraded once after the full gate passes; a non-empty current marker is
immutable. `droid_download_marker_selftest.json` records eight controls covering
the positive write/verify path and rejection of marker tampering, wrong routes,
ledger drift, count drift, and a same-size content replacement whose `mtime`
has been restored. The last control must still fail through the live `ctime`
and checksum-ledger binding.
After the immutable handoff, `qtail_verify_droid_mirror.py` independently
requires the exact official source, 4,102 objects, 4,096 TFRecords, the exact
3,700,745,265,151-byte total, matching object generations, a one-to-one MD5
ledger, no extra or partial files, and a successful `gsutil rsync -c` return
code. `droid_mirror_verifier_selftest.json` records eight tiny-fixture
controls. It accepts one valid fixture and rejects a nonzero checksum return
code, a missing object, same-size content drift with restored `mtime`, ledger
generation drift, a duplicate manifest path, an extra file, and the wrong
TFRecord count. These fixtures test fail-closed behavior; they do not replace
the formal 3.700 TB audit.
`droid_live_partial_marker_rejection.json` separately runs the production
verifier against the current incomplete official mirror. It passes only when
the verifier returns nonzero and creates no marker. This is real gate evidence,
but it is explicitly not full-mirror or formal-training evidence.

The downloader also has a process-level, nonblocking `flock` single-writer
contract. `droid_downloader_single_writer_selftest.json` records eleven
controls covering writer exclusion, lock-owner reporting,
capacity accounting, immutable verified objects, and rejection of unsafe
partial states. This code was added without restarting the active download
generation, so the live page must describe it as activating on the next natural
recovery or generation handoff. The controls prove downloader integrity, not
download completeness or model quality.

Per-release milestones keep transport and dataset-metadata byte semantics
separate. The 2,048 GCS TFRecord object sizes are sealed from the official
checksum manifest; `dataset_info.json` `split_bytes` is recorded alongside it
as a distinct metadata field. The DROID 1.0.1 zero-byte
`1.0.1_$folder$` object is assigned to that release's 2,051-object closure
without treating it as a TFRecord.
Before closure, each milestone URL serves a mutable `waiting` placeholder so
the progress evidence remains inspectable. The only permitted transition is
`waiting/immutable=false` to `complete/immutable=true`; after completion, any
change to counts, checks, or subset digests fails the milestone sealer. Here
`immutable=true` denotes that application-level stable-field contract, not a
filesystem flag or third-party timestamp.

Interrupted or replaced shard identities may leave older cache files in the
cache directory. They are preserved for audit but are not model inputs. The
cache manifest lists the exact selected artifacts and separately records the
directory count, unreferenced count/bytes, and a deterministic digest of
unreferenced cache names. The verifier recomputes those values before formal
training. After model fitting it independently streams every source TFRecord
again and recomputes every cached feature value. Formal completion requires
4,096/4,096 recomputed feature rows within deterministic floating-point
tolerance; a cache whose values were edited is rejected even if a new cache
manifest was generated for it.

The row contains trajectory length, reward summary, action statistics,
terminal rate, instruction complexity, and SHA-256 instruction fingerprints.
Raw instructions are not written to the feature cache.

## Tail taxonomy

The `record_informed_tail_v2` score combines:

- cross-shard instruction rarity;
- reward failure proxy;
- trajectory duration;
- action complexity;
- instruction complexity;
- episode-count rarity.

Components without variance are disabled and remaining weights are
renormalized. Shard number and shard position never contribute to the score.
Tail is the top 30% and extreme tail is the top 10% within the held-out split.

## PT allocation

The empirical probability column is loaded from the immutable PT source.
Positive finite probabilities are rank-quantile resampled to the number of
shards. The largest PT weights are mapped to the highest tail scores, then
blended as:

`Q-Tail target = 0.28 * source share + 0.72 * PT-tail share`

The source target is proportional to TFRecord bytes.

## Held-out evaluation

Each DROID release is split independently using SHA-256 of:

`seed || official release-relative TFRecord path`

The immutable seed is 11. The requested split is 80% model fitting and 20%
held-out evaluation; deterministic rounding yields exactly 410 held-out shards
per 2,048-shard release, 820 held-out shards total. The sorted official
relative-path membership is stored in the report and locked by SHA-256
`16781c97f05cc2bdc94837b0ae96942ac9621174d60775d2c6185dae5fd8a767`;
absolute mount paths cannot change membership. Feature
normalization, component min-max transforms, instruction document frequency,
tail-component activation, and PT allocation ranking are fit only on training
shards. Those frozen transforms are then applied to held-out shards.
Evaluation models never fit on held-out shards.

The separate deployment heads refit transforms and allocation targets on all
4,096 shards after held-out evaluation is fixed. Evaluation and deployment
normalization contracts are both stored in the model artifact.

### Rare instruction-fingerprint discovery

The report also includes an auxiliary breadth metric. Each raw instruction
sample is represented only by its SHA-256 byte fingerprint. Training-shard
document frequency is fit on the 3,276 training shards; a held-out fingerprint
is classified as rare when that training frequency is at most one. No
held-out instruction changes this rarity fit.

For each allocation head and each rare fingerprint, `p_fingerprint` is the
sum of held-out allocation probability over shards containing that fingerprint.
The expected distinct-fingerprint coverage after `B` independent,
with-replacement shard draws is computed exactly as:

`mean_fingerprint(1 - (1 - p_fingerprint) ** B)`

The immutable budgets are 10, 25, 50, 100, 200, 400, and 800 draws. The report
also gives the minimum draws needed to reach 10%, 25%, 50%, and 75% expected
coverage. The full curve is reported even when Q-Tail is faster at early
discovery but slower at broad late coverage. This metric is descriptive and is
not a completion gate.

If no held-out fingerprint satisfies the locked rarity threshold, the artifact
reports `no_eligible_fingerprints`, zero eligible/unseen counts, empty curve and
time-to-coverage arrays, and an explicit reason. This is an auxiliary `N/A`
result; it neither aborts formal training nor manufactures a coverage claim.

The fingerprints are byte-level equality proxies, not semantic task clusters.
Therefore `droid_rare_instruction_fingerprint_coverage.json` does not prove
semantic rare-task coverage, tail success, or robot-policy success. The final
pipeline independently reconstructs the split and predictions from the
structured training rows, recomputes the complete curve, and requires exact
agreement with both the standalone artifact and the training report.

## Equal-compute contract

The pinned official `droid_policy_learning` checkout is an immutable
reproducibility reference and is not invoked by this AllocationHead
experiment. The optimizer executions described below belong to the Q-Tail
AllocationHead implementation in this repository. Training or evaluating a
DROID robot policy with the official backend is a separate experiment and is
required before making any robot-policy or environment-rollout claim.

Both arms use:

- `AllocationHead(10 -> 32 -> 16 -> 1)`;
- AdamW, learning rate 0.002, weight decay 0.0001;
- the same seed, device, runtime-environment fingerprint, formal checkpoint-
  environment fingerprint, features, parameter count, and checkpoints;
- 20,000 evaluation-training steps;
- 20,000 all-shard deployment-training steps;
- 40,000 total steps per arm.

Here a step is not merely a requested loop index. Step 0 is the initialized
model, and checkpoint step `k` is the model and optimizer state after exactly
`k` calls to `optimizer.step()`. Therefore each arm must prove exactly 20,000
optimizer updates in evaluation training plus 20,000 optimizer updates in
deployment training, or 40,000 actual optimizer updates per arm. Every
per-stage resume audit records the target step, completed optimizer-update
count, training signature, and checkpoint boundary. A deterministic control
also verifies that resuming a 7-update fixture from checkpoint 3 is
bit-for-bit identical to an uninterrupted 7-update run. Checkpoints are
written through a temporary file and atomic rename. Checkpoint format v6
recomputes content hashes for the complete model tensor tree and optimizer
state tree and binds every later checkpoint to the previous checkpoint file
SHA-256. Truncated, out-of-range, wrong-device, wrong-optimizer,
model-tensor-mutated, and optimizer-moment-mutated checkpoints are required
negative controls.
The runtime-environment contract hashes hardware model, CPU architecture,
operating system, Python, PyTorch, MPS availability, and selected training
device. Checkpoint format v6 wraps that runtime contract with the SHA-256 of
the formal environment manifest, the checked live-code aggregate, the atomic
ORICO orchestration-snapshot manifest, and its code-parity result. The wrapped
checkpoint-environment fingerprint is part of the training signature, every
checkpoint, every resume audit, and the intermediate-checkpoint manifest. A
checkpoint from the same machine and software runtime but a different formal
code snapshot is rejected and training restarts at optimizer step 0. The
protocol self-test exercises this exact negative control.
The formal run must also produce an exact `4 x 5` checkpoint grid: evaluation
Source, evaluation Q-Tail, deployment Source, and deployment Q-Tail at
optimizer-update steps `0`, `5,000`, `10,000`, `15,000`, and `20,000`.
`droid_intermediate_checkpoint_manifest.json` records every checkpoint path,
byte count, SHA-256, stage, update count, device, optimizer, seed, and training
signature. It also records content-derived input-feature, initialized-state,
current model-state, and optimizer-state fingerprints plus the parent
checkpoint name, step, and SHA-256. Evaluation Source and
Q-Tail must have the same feature fingerprint, deployment Source and Q-Tail
must have the same feature fingerprint, and all four step-0 model states must
have the same initialized-state fingerprint. The step-0 fingerprint is
recomputed directly from checkpoint tensors. A missing, extra, unreadable,
renamed, fingerprint-mutated, or semantically mismatched checkpoint
invalidates the training-complete marker.

Before formal model fitting, the pipeline writes
`droid_environment_manifest.json`. It independently records the Python,
PyTorch, NumPy and MPS runtime, macOS and hardware facts, ORICO mount, DROID
backend commit, installed package versions, immutable-input hashes, critical
code hashes, and this exact compute contract. Only an explicit deterministic
environment-variable allowlist is included; proxy URLs and credentials are
excluded. The environment manifest must pass every gate and is itself included
in the final artifact SHA-256 manifest. Its black-box controls also require a
clean cumulative UniClash isolation record to pass and a fixture containing a
single blocked/proxied transfer to fail.

Formal mode is locked before feature extraction: it requires the verified
4,102-object mirror, all 4,096 TFRecords, the exact official 187,891-record
count, `record_parse_rate=1.0`, and `record_scan_complete_rate=1.0`. A partial
mirror, sampled record cap, relaxed parse threshold, or mismatched official
record count cannot enter formal training.

The outer pipeline invocation is separately locked by
`droid_pipeline_shell_contract_selftest.json`. Its nine controls require the
actual trainer call to retain the verified-mirror flag, ORICO mount, full-record
mode, 20,000 steps, 5,000-step checkpoints, seed 11, 5,000 bootstrap samples,
the 20% holdout, and every bound manifest/ledger input. The same source check
requires monotonic checksum, environment capture, prewarm exit, formal
training, full cache recomputation, record closure, training-marker, final QA,
process-log seal, and public-projection order. Mutations that remove the formal
launch, verified-mirror gate, publication commit, parent ownership, or
read-only completed-state handling must fail.

The training-start marker is written only after those exact mirror and record
closure checks, 100% parse/scan checks, and the deterministic release-stratified
holdout count check have passed. `droid_training_gate_order_selftest.json`
seals eight controls: one positive control over the current trainer and seven
source mutations that move the mirror, record closure, parse, scan, holdout,
training-marker, or optimizer boundary into an invalid order. Any failed
control withholds formal training before the first optimizer-backed stage.

Final page QA also copies the eight required live process logs into
`process_logs_final/` and writes `droid_process_log_manifest.json`. Each log
snapshot records its source, role, byte count, line count, and SHA-256. The
seventh required log is `manual_endpoint_generation_handoff.log`, which binds
the bounded endpoint/HTTP/worker controls and both production handoffs.
The eighth is `qtail-web-services.log`, which records dual-port page
supervision and recovery. The minute progress loop atomically mirrors six
local supervision streams into `ORICO/qtail_full_training/logs/` so they do
not depend on the workstation `.tmp` lifetime. Together with the watchdog
status already written on ORICO, the same snapshot retains seven non-gating
supervision artifacts: the watchdog status, scheduled launcher log, both
launcher stdout/stderr streams, both UniClash guard stdout/stderr streams,
and the local web-service log. These histories are hashed and listed but
cannot replace any of the eight required logs.
The final artifact gate requires the manifest and every required snapshot, while
the original live logs remain available through `live_logs/` for
post-completion supervision. The mutable `live_logs/` targets are explicitly
excluded from the final artifact manifest. Final marker contract
`droid_final_page_qa_marker_v2` binds the process-log manifest directly; the
independent marker verifier recomputes every required log hash through the
artifact manifest before accepting a resumed completed run.

Final page QA and the independent final stage-marker verifier also require the
timeline verification artifact to retain a structured data-continuity summary:
authoritative completed-object and official-MD5-ledger decrease counts,
checksum-error sample count, legacy physical-byte cleanup events, feature scan
pass resets, committed-feature-counter decreases, and the associated claim
boundary. Repair events remain visible rather than being rewritten as a
monotonic history. Historical scan-pass resets remain disclosed because an
in-progress cache scan legitimately restarted at shard zero. From the
`monotonic_committed_prewarm_snapshot_v1` activation onward, the public
preparsed-shard and record counters come only from the last atomically committed
prewarm pass and are required never to decrease. The active scan pass remains a
separate transient field. A controlled self-test requires legacy reset
disclosure to pass, committed-counter decrease to fail, and committed growth to
pass.

Runtime health also audits reboot supervision on every status refresh. Both
user LaunchAgents must be loaded, their installed plists must be byte-identical
to the repository copies, the pipeline launcher must retain its 300-second
scheduled retry, and the UniClash transport guard must retain `KeepAlive`,
two-second sampling, `en1`, and `Wi-Fi` bindings. This verifies supervision
configuration, not future electricity, network, or ORICO availability.

Before the multi-terabyte mirror closes, a separate bounded engineering
preflight selects four checksum-ledger-verified TFRecords from each official
release, recomputes all eight local MD5 values against the official values, and
decodes exactly two records per shard. It runs Source and Q-Tail on MPS with
the same architecture, AdamW configuration, seed, features, parameter count,
runtime-environment fingerprint, and 50 optimizer updates per arm. A second
identical invocation must resume all four stages from step 25 without changing
any of the 16 checkpoint-v6 parent-chain hashes. Its current-generation
summary is `droid_preflight_training_smoke.json`; the unique run directory
retains the frozen shard list, first/resume logs, first/final reports,
checkpoints, and hashes. The trainer receives no marker directory, and the
runner requires every formal marker to remain unchanged. The preflight is
explicitly test-only: a passed or failed tiny-sample hypothesis gate is
excluded from the formal claim and can never create a training-complete
marker.

An additional predictive engineering run freezes the first 908 fully scanned
DROID 1.0.0 shards (40,686 records) and uses the formal 20,000-update
per-stage budget. It exists to detect model or protocol failures before the
multi-terabyte mirror closes. Its sealed summary is
`droid_forecast_908_summary.json`; `formal_protocol.locked` remains false
because DROID 1.0.1 and the complete 4,096-shard mirror are absent. This run
is summarized reproducibly by `tools/qtail_summarize_droid_forecast.py` and
observed a +11.75 pp held-out tail-allocation-share forecast and a +26.32 pp
extreme-underallocation reduction, while rare instruction-fingerprint
coverage was slower at every reported draw budget. Both directions must be
shown together. None of these forecast values can satisfy the formal
completion gate or be presented as robot-policy tail success.

## Statistical decision

The reported tail-allocation-share gain is computed only on held-out
predictions from the allocation head. It is a mechanism-consistency result:
the Q-Tail target and the evaluation tail mask use the same frozen,
training-fit tail taxonomy. It is not an independent causal test and is not
robot-policy success, environment rollout success, or a replacement for
policy training.

Uncertainty uses 5,000 paired, release-stratified shard bootstrap samples, so
every draw preserves the held-out count from each official DROID release.
Source and Q-Tail weights are renormalized inside each draw. The bootstrap
reports a percentile interval and a descriptive fraction of replicates at or
below zero. That fraction is explicitly not labeled or interpreted as a
hypothesis-test p value.

A separate 5,000-sample paired shard arm-swap diagnostic swaps Source and
Q-Tail labels within each held-out shard pair and renormalizes both arms
inside every swap. Because allocation weights are coupled by a global softmax
normalization, shard pairs are not independent exchangeable units. The
resulting nonpositive fraction is therefore descriptive sensitivity evidence,
not a valid hypothesis-test p value, and is not a completion or support gate.

The preregistered outcome is `supported` only when all are true:

- held-out tail-allocation-share gain is at least 2 percentage points;
- stratified 95% CI lower bound is at least 2 percentage points;
- extreme-tail underallocation reduction is positive.

It is `not_supported` when the 95% CI upper bound is below 2 percentage points
or extreme-tail underallocation reduction is nonpositive; all remaining valid
executions are `inconclusive`. A complete, publishable formal experiment may report any of
these three outcomes. Project completion depends on execution validity,
artifact integrity, and reproducibility, never on obtaining a favorable
result.

The held-out unit is an official TFRecord shard, stratified by release. This
does not prove task-, scene-, collector-, or instruction-fingerprint-isolated
generalization. That limitation is retained in the report and page claim
boundary.

## Technical and commercial decision boundary

The live page keeps the technical and commercial interpretation mechanically
coupled to the publishable formal outcome. Before the verified full mirror,
record closure, same-compute execution, `DROID_TRAINING_COMPLETE`, final page
QA, and the validated `DROID_PUBLIC_PROJECTION_COMMITTED` seal all pass, the
section must remain `WITHHELD`, `waiting for formal decision`, and `no effect
claim`. Training completion proves execution validity but is not publication
authority. Forecast and scalability-canary values are never inputs to this
state.

After publication, the three preregistered outcomes map as follows:

- `supported`: the allocation-head tail objective is supported and the system
  may enter a bounded customer data-curation/allocation pilot;
- `inconclusive`: the evidence remains uncertain and only an exploratory,
  preregistered customer pilot is justified;
- `not_supported`: the preregistered target is not supported and no uplift
  sales claim is permitted.

All three states retain the same boundary: the evidence concerns DROID
AllocationHead budget redistribution, not robot-policy success, environment
rollout, revenue, ROI, or deployment guarantees. The browser smoke verifier
runs the live incomplete page on desktop and mobile and separately exercises
all three formal mappings with presentation-only fixtures. The fixture scope
is explicitly `presentation_logic_only_not_formal_training_evidence` and
cannot create or satisfy any formal result or completion marker.

`DROID_TRAINING_COMPLETE` is a structured, atomic marker rather than an empty
sentinel. It binds the verified mirror, official two-release metadata audit,
protocol controls, environment contract, independently recomputed
4,096-shard cache, final report, model status, and checkpoint by exact path,
byte count, and SHA-256. It also binds the v2 incremental closure, its 7/7
controls, the two-release milestone status, and both immutable per-release
milestone files. The semantic verifier requires exact environment contract v3
controls 9/9,
zero deferred or missing TFRecords, and exact 4,102-object / 4,096-shard /
187,891-record closure. Every pipeline start revalidates those bindings and
the formal semantics; a stale or edited artifact invalidates the training
marker and its dependent page-QA marker before any stage can be skipped.
Immediately before committing that marker, the artifact-manifest merger
re-reads and re-hashes every retained artifact and adds the v2 closure,
closure controls, release status, both immutable release milestones, the
current transport-classifier control, its preserved coverage-gap epoch, and
the disclosed 908-shard forecast summary. When present, the bounded
2,505-shard canary summary, sealed full report, sealed frozen membership list,
and shard-list controls are retained and re-hashed as optional historical
evidence. They are explicitly excluded from the formal required-path set and
cannot satisfy a full-data completion gate. The manifest merger's 7/7
positive and negative controls verify optional-history retention and hashing,
missing-history pruning, required-set drift rejection, missing formal
artifacts, symlink escape rejection, and path traversal rejection. Formal mode
explicitly requires all 64 pre-page artifacts: 42 static artifacts, the full
20-checkpoint grid, and two immutable release milestones. One missing file
withholds the training marker.

The static shell contract is versioned
`droid_pipeline_shell_contract_v9`. Its eleven controls cover the direct-route
download invocation, verified-mirror training gate, terminal event order,
two-phase public projection, post-commit read-only QA, parent ownership, and
the final process-log snapshot. It additionally requires the snapshot-bound
environment controls/capture and exactly one environment manifest on the
formal trainer command. It also parses the release-milestone command
block and requires the formal DROID data root to be supplied exactly once, so
duplicated or ambiguous milestone inputs fail before the full run advances.

The generation handoff polls the immutable download marker every second.
Existence alone is insufficient: it first reruns the official manifest, MD5
ledger, transport-status, exact byte-total, object-count, TFRecord-count, and
live-file binding verifier. Only a valid marker allows the old pipeline shell
to be stopped before its descendants are
terminated, limiting the old-generation checksum launch window and ensuring
the continuously guarded checksum stage is entered by the current code.
While that marker is absent, the watchdog also requires one handoff command
bound to the current pipeline PID and recreates it if missing. This makes the
binding self-healing after a reboot or watchdog restart. Auto-launch is
disabled after `DROID_DOWNLOAD_COMPLETE` exists, so checksum and training
cannot enter a handoff restart loop.

Every current-generation pipeline writes an atomic `PIPELINE_STARTED` marker
with format `qtail_pipeline_started_marker_v2`. It binds the live pipeline PID,
exact script path and SHA-256, ORICO job root, lock path, and lock-owner PID.
Before checksum, environment capture, and formal training, the pipeline runs
the same nine-check semantic generation gate and atomically records all three
results in `pipeline_generation_gate.json`. The final gate must be
`pre-formal-training`; all three gates must share one live PID and one script
hash, and that hash must match both the marker and current source. A legacy
pipeline may remain alive only during resumable download while a uniquely
bound generation-handoff watcher is healthy. Formal training and final QA
require `HASH MATCH`; a stale, empty, edited, or PID-mismatched marker exits
before the irreversible stage so the watchdog can start the current source.
The training-complete marker directly binds this generation report and its
semantic verifier requires the ordered three-gate history, one PID, one
current source hash, all nine checks per gate, and the exact pipeline command.
Four destructive controls reject a false check, a missing gate, and
post-gate source drift in addition to the positive three-gate case.

## Completion contract

The run is complete only after all nine independently audited requirements
pass:

1. existing Open X assets and the pinned training backend are on ORICO;
2. the official DROID source, immutable object/checksum manifests, and exact
   two-release metadata/schema audit match;
3. UniClashCore remains online while DROID transfers pass the direct-route
   isolation guard;
4. all 4,102 objects and 3,700,745,265,151 bytes pass MD5 and mirror closure;
5. all 4,096 TFRecord shards and all decodable records pass the record audit;
6. Source and Q-Tail finish the equal-compute training contract;
7. intermediate artifacts, checkpoints, environment evidence, and SHA-256
   manifests verify;
8. the background runtime and served evidence endpoints remain healthy;
9. the final page passes desktop and mobile evidence QA.

The live completion state is stored in `completion_audit.json`. A successful
pipeline stage cannot substitute for any missing requirement.

The complete process history is stored in `pipeline_timeline.json`. On first
creation it imports every retained sample from
`download_progress_samples.json`, then records a full pipeline sample on every
status refresh. Full samples include physical and completed bytes, object
checksums, feature-shard and record counts, model stage and optimizer progress,
ORICO capacity and sleep policy, the `ExternalMedia` assertion,
UniClashCore/TUN/direct-route counters, process invariants, committed markers,
and the nine-gate completion count. Every entry carries its sequence number,
the previous entry hash, and its own canonical SHA-256. The
independent timeline verifier rejects reordered, removed, inserted, or edited
samples. Its `data_continuity` summary separately counts completed-object and
official-MD5-ledger decreases. Those are the authoritative persistence
signals. Legacy physical-byte decreases are retained and counted but can come
from temporary-part cleanup during the original gsutil-to-HTTPS transition.
Historical feature extraction used the repeated cache scan's transient
processed-shard counter, so it could restart from zero without implying that
verified cache files or DROID objects were lost. Those events remain in the
chain. New samples publish the last complete prewarm snapshot as the monotonic
cumulative counter and retain current-pass progress separately; a decrease in
the committed counter now invalidates timeline verification.

During the download stage, runtime health also requires exactly one generation
handoff watcher. Its command-line target PID must equal the current unique
pipeline PID. When the immutable download marker appears, that watcher freezes
the old pipeline before it can enter checksum or training, removes its
descendants, and lets the watchdog start the current code generation. A stale,
missing, duplicate, or misbound watcher therefore closes the runtime gate
before the 100% transition.

The scheduled launcher also treats process existence and activity as separate
conditions. Progress must refresh `latest.json`, the watchdog must refresh its
atomic status file, and prewarm must either have caught up to the complete
TFRecord count or publish a fresh atomic liveness heartbeat. That heartbeat
binds the unique prewarm shell PID, current phase, observed complete-shard
count, and active child PID. Long feature, record-audit, closure, self-test, and
release-milestone phases refresh it every 20 seconds; the idle loop refreshes it
every 60 seconds. A heartbeat from a prior or second prewarm process is
rejected. A stale or duplicate sidecar is replaced as a process tree, so a
detached screen shell cannot leave an orphaned parser or verifier behind. This
repairs both the observed dead-loop failure mode and the later false restart
mode where a healthy idle loop was killed every five minutes because no new
feature-status row was expected while it slept.

`droid_runtime_process_contract_selftest.json` supplies sixteen controlled
checks. In addition to the valid and destructive topology, heartbeat, exact
PID ownership, launcher, mount, cleanup, web-service, handoff-marker, and
Python-script classification controls, v11 requires an atomic ORICO write
probe before the scheduled launcher may stop or replace any worker. This
rejects the observed macOS privacy-domain failure in which a LaunchAgent could
see ORICO, kill a Terminal-owned prewarm loop, and then fail to write its own
replacement. Version 11 also requires the exact symlink-enabled web command
and rejects a mutation that removes `--symlinks`, preventing a healthy-looking
page from publishing READY artifact links that return 404. The same v11 suite
proves the post-download convergence order.
The handoff must start the current watchdog before waiting for prewarm;
the watchdog must start a missing current-generation pipeline; that pipeline
must commit the checksum marker before waiting for prewarm; and prewarm must
exit naturally on that marker. Four negative mutations remove each edge and
must all be rejected. Merely mentioning a guard or downloader path in a
diagnostic shell command is not a live process match. The production pipeline
reruns these controls on every natural generation start, while live identity
is evaluated from the real process table each minute. The independently loaded
transport guard keeps UniClashCore online while terminating only DROID transfer
processes that lose the explicit direct-route contract.

The formal environment seal also treats the official training backend as an
immutable input. It requires origin
`https://github.com/droid-dataset/droid_policy_learning`, commit
`9a29c832b4c81bf38401111f5e4cdddaca217581`, a clean tracked and untracked
worktree, and a successful `git fsck`. Environment contract v3 runs nine
black-box controls: the original positive, byte-mismatch, missing-MD5,
UniClash-violation, transport-classifier, backend commit, origin, and untracked
worktree controls, plus orchestration-snapshot code drift. Every mutation must
fail before formal training can start.

The live runtime projection also records the macOS AC `sleep` and `disksleep`
values plus the `ExternalMedia` power assertion. Runtime health requires
`sleep=0`, `disksleep=0`, the real ORICO mount, and an active external-media
assertion. This is a live cross-day execution gate; it does not guarantee
utility power, USB cable integrity, network availability, or future mount
availability.

The timeline keeps runtime process-count anomalies separate from route
evidence. Such a sample remains visible and its runtime state may be unhealthy,
but a fresh guard heartbeat with zero forbidden sockets and zero wrong routes
is not relabeled as VPN traffic. The verifier reports both anomaly counts and
VPN-route violation counts.

The verifier also emits `transport_evidence_scope`. It measures the interval
between the first retained download sample and the first hash-preserved guard
sample, reports the observed filesystem-byte change in that interval, counts
known classifier coverage-gap epochs, and lists inter-epoch gaps. Those values
are visible on the live page. They are disclosure boundaries, not inferred VPN
events: pre-guard bytes have no retrospective route evidence, and sampled
sockets do not prove the absence of unobserved traffic.

Final page QA uses a lease-bound bootstrap followed by a full marker commit.
`FINAL_PAGE_QA_PREVIEW` names the parent pipeline PID and an expiry time; an
orphaned or expired lease is ignored. A bootstrap marker with status `sealing`
binds only already-immutable training outputs and is effective only while that
lease remains valid. This breaks the fixed-point dependency between rendering
and sealing the QA evidence without allowing a preview marker to publish
completion.

While the bootstrap is active, the verifier writes the QA JSON, desktop/mobile
screenshots, immutable download history, final hash-chained timeline, and final
timeline verification against the honest precommit `8/9` state, probes every
displayed artifact link, and merges their SHA-256 values. Missing future-stage
links must render as `WAIT`. A present file becomes `READY` only when it is
nonempty and any stage-specific semantic state is valid; in particular, failed
final-QA JSON and its screenshots remain `WAIT`. The parent pipeline hashes the
successful QA JSON, records
`QTAIL_TERMINAL qa_sealing_complete qa_sha256=<sha256>`, refreshes immutable
process-log snapshots, and atomically replaces the bootstrap with the fully
bound committed marker. The later `qa_commit_complete` event is an audit event
after publication, not a prerequisite that would make the first seal
impossible. The preview lease is removed only after the full marker verifies.
A crash or failed check removes the bootstrap and causes the public state to
return to 8/9.

The committed final marker binds the training marker, final artifact
manifest, immutable download history, immutable hash-chained pipeline
timeline, its independent final verification, immutable UniClash guard, QA
JSON, and both screenshots by byte count and SHA-256. The live status builder
accepts a bootstrap only with a live pipeline lease and accepts a committed
marker only after all bindings rehash successfully. Marker presence alone can
never move the page to `complete`. Live `latest.json`,
`completion_audit.json`, progress samples, and the current timeline remain
semantic gates but are not hash-bound because the progress loop refreshes them;
their immutable final counterparts are bound in a second phase. After the full
marker replaces the bootstrap, the status builder performs one final projection
refresh. `DROID_PUBLIC_PROJECTION_COMMITTED` then binds the final QA marker,
the live `latest.json` and `completion_audit.json`, and byte-identical
`latest_final.json` and `completion_audit_final.json` snapshots. The complete
final verifier rejects any live/snapshot divergence. Public state freezes only
when the evidence says `committed=true`, `preview=false`, and
`qa_state=committed`.

Final publication holds the progress refresh lock while building the candidate
9/9 projection, writes immutable snapshots and
`DROID_PUBLIC_PROJECTION_COMMITTED`, then publishes
`completion_audit.json` before publishing `latest.json` last. The final marker
alone therefore remains an honest 8/9 sealed state; no observable live
projection may claim 9/9 without a valid second marker. A non-smoke page
verifier that encounters an already-complete state is read-only, and a
precommit verifier must be owned by the live parent pipeline.

Postcommit browser QA also has an exclusive run lock. A second live QA process
is rejected before Chrome starts, while a stale lock owned by a dead PID is
recoverable. The postcommit marker commit independently holds an advisory file
lock across existing-marker validation, artifact validation, and atomic write.
An already-valid marker is returned byte-for-byte without refreshing its
commit timestamp, so concurrent or resumed callers cannot reseal the same
evidence as a new completion event.

`DROID_POSTCOMMIT_PAGE_QA_COMPLETE` binds seven paths by byte count and
SHA-256: the committed final marker, committed public-projection marker, frozen
`latest_final.json`, frozen `completion_audit_final.json`, the read-only browser
QA report, and its desktop and mobile screenshots. These postcommit artifacts
are deliberately outside the 64-file pre-page formal manifest. Requiring them
before the public 9/9 projection would create a circular gate because their
browser contract must first observe that committed 9/9 projection. The final
stage validator requires both layers and rehashes the postcommit marker after
the browser run, so the two-stage design does not weaken final evidence.

The marker hardening suite contains 38 controls. Five controls independently
enforce the exact seven-case incremental-closure contract: the canonical
payload passes; a missing formal gate check, an extra check, a false check, or
a formal rejection case spoofed as success must fail. Four pipeline-generation
controls require the ordered pre-checksum, pre-environment, and
pre-formal-training gates, and reject a false check, an incomplete gate
history, or source drift. Two publication-boundary controls require an honest
8/9 projection to keep formal values withheld and reject a training-only
publication flag. One identity-contract control requires the canonical
38-name set and rejects a missing, extra, or duplicate control. Eight terminal
integration controls execute the actual
lease and projection validators in a temporary job root: rejection
without a bootstrap, rejection of an expired lease, rejection of a
lease-owner mismatch, an honest 8/9 final-marker commit, the atomic 8-to-9
dual-marker commit, byte identity between live and frozen projections,
rejection of a modified frozen snapshot, and rejection of a modified final
marker. Heavy mirror, model, manifest, continuity, and process-log semantics
are replaced only at their ownership boundary so the test exercises the real
public-state transition rather than fabricating a completed DROID run.

The progress projection reports a checkpoint as `SAVED` only when the exact
expected nonempty file exists. It reports `VERIFIED` only when the manifest
path, byte count, SHA-256, step, optimizer-update count, stage, and target all
match and the final training marker is independently valid. Model-step text
alone can never color a checkpoint cell as saved or verified.

The projection suite contains 15 controls. Four artifact controls use real
temporary files to require a passing JSON to become `READY`, reject failed and
malformed JSON, keep the failed final-QA JSON and both screenshots at `WAIT`,
and admit that three-file family only after QA status is complete. Live
`in_progress`, `waiting`, or `recording` documents remain readable evidence;
explicit failed, error, invalid, blocked, unavailable, `valid=false`, or
`passed=false` documents do not.

Three artifact-count controls additionally prove that the 64-item pre-page
formal baseline remains fixed while effective QA adds nine immutable process
log artifacts (the manifest plus eight required logs) and committed QA adds
five final browser/transport artifacts. The dynamic totals become 73 and 78
without changing the reader-facing baseline.
The public completion projection also carries the sorted, stable
`required_artifacts` path set, not only the currently missing subset. The page
renders every path in a formal artifact contract ledger as `SEALED`,
`GENERATED`, or `WAIT`. Browser QA requires exact row/count agreement, unique
relative links, all 20 checkpoint paths, and the checksum-handoff,
pre-environment, and pre-training gates at both desktop and 390-pixel mobile
widths. Every `GENERATED` or `SEALED` ledger link must also return HTTP 200;
`WAIT` paths remain unrequested until their owning stage creates them. This
keeps the full contract visible after sealing reduces the missing list to zero.
The fifteenth control also proves that the ORICO snapshot manifest accepts an
identical workspace source, rejects source-content drift, and rejects a
source-relative path that escapes the workspace. Snapshot self-consistency
alone therefore cannot satisfy the migration requirement.

The 9-control shell contract also names every pipeline-owned producer needed
for the 42 static formal artifacts: manifest/checksum builders, release audit,
timeline/protocol/environment controls, the all-record trainer, cache and
closure verifiers, release milestone sealer, artifact merger, transport gates,
and destructive self-tests. Removing a producer therefore withholds execution
before the irreversible training or final-publication transitions.

Formal optimizer startup has a three-stage code-generation binding. The
pipeline start marker binds the running shell PID, lock owner, and shell hash;
environment capture requires every critical workspace hash to equal the
atomically published ORICO orchestration snapshot; and the trainer re-hashes
the complete environment inventory plus snapshot manifest before verified
mirror parsing or any optimizer update. The 11/11 gate-order suite rejects
misordering and live code drift, and the training/final judges repeat the
binding through publication.

These SHA-256 bindings prove internal consistency against the retained files;
they are not an external timestamp, transparency log, or WORM guarantee. An
external Git commit/PR or independent timestamp receipt is a separate
publication step and must not be inferred from local hashes alone.

Training artifacts follow the same sealing rule: model status, run manifest,
training summary, and report are finalized before their byte counts and
SHA-256 values enter `droid_artifact_manifest.json`. The final verifier rejects
any post-manifest mutation and requires the complete named evidence set,
including the final checkpoint, training rows/curve, cache verification,
rare instruction-fingerprint curve, protocol controls, environment evidence,
the exact 20-entry intermediate-checkpoint manifest, final UniClash snapshot,
QA JSON, and both screenshots. Merely listing arbitrary files cannot satisfy
the artifact gate.

## Reproduction

Bounded engineering canaries may use `--shard-list` with a sorted,
hash-bound relative-path manifest so membership remains stable while the
official mirror continues downloading. This option is test-only, mutually
exclusive with `--max-shards`, and forces `formal_run=false`; the immutable
formal protocol never accepts a frozen subset list.

### Frozen scalability canary (non-formal)

The retained engineering canary freezes 2,505 complete shards with membership
digest
`ed7538ed6d30c17098fcb9c984de03b084215c41c51f2a852b8e21087e27472a`.
It decodes 113,426 records (DROID 1.0.0 complete and DROID 1.0.1 partial),
then runs four MPS AllocationHead stages with 1,000 optimizer updates each.
Source and Q-Tail receive 2,000 updates per arm with the same architecture,
optimizer, seed, features, parameter count, device, and runtime fingerprint.

All four stages resumed from step 1,000 without rejection. The 20 intermediate
checkpoint hashes, parent chains, and final model hash remained byte-identical
after same-source replay. Eleven shard-list controls pass, including rejection
of duplicates, unsorted membership, digest tampering, traversal, external
symlinks, partial downloads, and `--max-shards` conflicts, plus positive
controls that a frozen list preserves its record cap while an unbounded mirror
scan forces all-record mode.

The held-out allocation diagnostic is +14.23 percentage points
(32.76% to 47.00%; conditional bootstrap interval +12.76 to +15.40 pp), and
the extreme-underallocation diagnostic is +29.41 pp. The byte-level rare
instruction-fingerprint proxy is mixed and becomes negative at larger draw
budgets (-2.60 to +0.22 pp across reported budgets). These numbers are retained
for engineering diagnosis only. DROID 1.0.1 is incomplete, the input audit is
not a verified 4,096-shard mirror, and none of these values may be promoted to
the final claim or interpreted as robot-policy tail success.

Artifacts:

- `results/qtail_droid_full/droid_scalability_canary_summary.json`
- `results/qtail_droid_full/droid_shard_list_selftest.json`
- `results/qtail_droid_full/droid_scalability_canary_frozen_shard_list.json`
- `results/qtail_droid_full/droid_scalability_canary_full_report.json`

```bash
python3 tools/qtail_train_droid_full.py \
  --data-dir /Volumes/ORICO/qtail_full_training/data/droid \
  --out /Volumes/ORICO/qtail_full_training/results/qtail_droid_full \
  --marker-dir /Volumes/ORICO/qtail_full_training/markers \
  --object-manifest /Volumes/ORICO/qtail_full_training/results/qtail_droid_full/droid_object_manifest.json \
  --checksum-manifest /Volumes/ORICO/qtail_full_training/results/qtail_droid_full/droid_object_checksum_manifest.json \
  --checksum-ledger /Volumes/ORICO/qtail_full_training/results/qtail_droid_full/droid_object_checksum_ledger.json \
  --transport-status /Volumes/ORICO/qtail_full_training/results/qtail_droid_full/parallel_download_status.json \
  --download-marker /Volumes/ORICO/qtail_full_training/markers/DROID_DOWNLOAD_COMPLETE \
  --download-verification /Volumes/ORICO/qtail_full_training/results/qtail_droid_full/download_verification.json \
  --environment-manifest /Volumes/ORICO/qtail_full_training/results/qtail_droid_full/droid_environment_manifest.json \
  --require-verified-mirror \
  --required-mount /Volumes/ORICO \
  --records-per-shard 0 \
  --steps 20000 \
  --seed 11 \
  --holdout-fraction 0.20 \
  --bootstrap-samples 5000 \
  --pt-source data/uploaded_data.csv
```
