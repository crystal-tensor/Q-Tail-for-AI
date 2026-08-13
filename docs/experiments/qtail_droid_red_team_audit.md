# Q-Tail DROID Full Run Red-Team Audit

Date: 2026-07-29

## Scope

Two independent read-only reviewers audited the formal DROID trainer and the
download-to-training evidence pipeline while the official mirror continued to
download. The audit did not use the forecast result as formal evidence.

## Claim adjudication

The current experiment can test whether a Q-Tail allocation head assigns more
fixed probability mass to held-out shards that rank highly under the
pre-registered record-informed tail taxonomy. It cannot establish end-to-end
robot-policy tail success, semantic rare-task coverage, or real-world extreme
failure reduction.

The Q-Tail target and the allocation metric use the same pre-registered tail
taxonomy. A positive result is therefore mechanism-consistency evidence, not an
independent causal validation of the taxonomy or a robot policy.

## Trainer findings

### RT-01: Policy-success claim is out of scope

Severity: P0

`AllocationHead` predicts allocation mass. There is no policy rollout, action
success label, or environment success evaluation. Tail allocation share and
extreme underallocation must not be renamed to tail success or extreme robot
failure rate.

Evidence:

- `tools/qtail_train_droid_full.py:270`
- `tools/qtail_train_droid_full.py:1600`
- `tools/qtail_train_droid_full.py:1650`

Disposition: permanent claim boundary. A separate same-policy training and
environment-rollout experiment is required for policy-success claims.

### RT-02: Target and allocation metric share the tail taxonomy

Severity: P0

The PT allocation target is ordered by tail score, and the held-out allocation
metric selects the top tail-score shards. This can validate learning and
generalization of the allocation mechanism, but not an independent downstream
effect.

Evidence:

- `tools/qtail_train_openx_demo.py:391`
- `tools/qtail_train_openx_demo.py:444`
- `tools/qtail_train_droid_full.py:1585`

Disposition: permanent claim boundary shown in the page and final report.

### RT-03: Bootstrap nonpositive fraction was labeled as a p-value

Severity: P1

The release-stratified paired bootstrap describes sensitivity of fixed model
predictions to held-out shard composition. Its nonpositive replicate fraction
is not a null-hypothesis p-value and does not include training-seed or split
uncertainty.

Evidence:

- `tools/qtail_train_droid_full.py:709`
- `tools/qtail_train_droid_full.py:741`

Disposition: retain bootstrap confidence intervals as conditional descriptive
uncertainty. The later paired arm-swap calculation is also retained only as a
descriptive diagnostic because global softmax normalization violates its
exchangeability assumption; neither fraction is presented as a valid p value.

### RT-04: Rare coverage is an analytic fingerprint proxy

Severity: P1

The curve is the analytic expected discovery probability under independent
sampling. It is not observed generation, semantic clustering, policy training,
or task execution. A fingerprint with training document frequency one is rare
but not unseen.

Evidence:

- `tools/qtail_train_droid_full.py:569`
- `tools/qtail_train_droid_full.py:590`

Disposition: keep it auxiliary and descriptive; report seen-once and unseen
subsets separately when available.

### RT-05: Resume checkpoints did not bind the full environment

Severity: P1

Device and optimizer strings were checked, but hardware, OS, Python, PyTorch,
and MPS facts were not part of the checkpoint training signature.

Evidence:

- `tools/qtail_train_droid_full.py:122`
- `tools/qtail_train_droid_full.py:313`
- `tools/qtail_train_droid_full.py:1878`

Disposition: bind a stable secret-free environment fingerprint into training
signatures and checkpoints, and reject resume across mismatched environments.

### RT-06: Standalone trainer could relax full-record thresholds

Severity: P2

The production pipeline passed both scan thresholds as 1.0, but the trainer
defaults and formal argument lock did not independently require them or require
the official full record count.

Evidence:

- `tools/qtail_train_droid_full.py:1010`
- `tools/qtail_train_droid_full.py:1098`
- `tools/qtail_train_droid_full.py:1175`

Disposition: make the trainer fail closed for formal runs.

## Pipeline findings

### RT-07: QA preview could appear as formal completion

Severity: P1

A live `FINAL_PAGE_QA_PREVIEW` was accepted by public progress state as if the
immutable final marker already existed.

Evidence:

- `tools/qtail_droid_full_progress.py:2485`
- `tools/qtail_droid_full_progress.py:2603`

Disposition: preview may show QA in progress but can never satisfy a completion
requirement or set public status to complete.

### RT-08: UniClash continuity did not cover every handoff

Severity: P1

The active Core/TUN/direct-route gate surrounded checksum transfer but was not
rechecked at every cached-marker and training-start handoff.

Evidence:

- `scripts/qtail_orico_full_pipeline.sh:407`
- `scripts/qtail_orico_full_pipeline.sh:431`
- `scripts/qtail_orico_full_pipeline.sh:538`
- `tools/qtail_verify_droid_stage_markers.py:304`

Disposition: require a fresh Core ON, TUN OFF, direct-route guard before
environment capture and immediately before formal training, including resumed
checksum-marker paths.

### RT-09: Final marker did not recursively bind every artifact

Severity: P1

The final marker bound the artifact manifest file but did not recursively
recompute every manifest entry or bind the served page and live final data
source.

Evidence:

- `tools/qtail_verify_droid_stage_markers.py:121`
- `tools/qtail_verify_droid_stage_markers.py:867`
- `scripts/qtail_orico_full_pipeline.sh:1125`

Disposition: recursively verify path, byte count, and SHA-256 for every final
artifact; bind the final page source and immutable terminal snapshots. Live
status files remain semantic gates but are excluded from marker hashes because
the progress loop continues refreshing them. Stale markers must be invalidated
and QA rerun.

### RT-10: Historical prerequisites were not self-healing

Severity: P2

The final manifest required historical self-test and forecast artifacts that
the nominal post-download path did not regenerate when missing.

Evidence:

- `tools/qtail_merge_droid_artifact_manifest.py:15`
- `scripts/qtail_orico_full_pipeline.sh:640`

Disposition: regenerate reproducible controls or fail early with an explicit
repair step before expensive formal training.

### RT-11: Log closure checked hashes but not terminal events

Severity: P2

An empty or stale log snapshot could satisfy byte/hash consistency without
proving checksum closure, 187,891-record closure, training completion, and QA
commit.

Evidence:

- `tools/qtail_verify_droid_stage_markers.py:137`

Disposition: require nonempty snapshots, named terminal events, and timestamps
not earlier than the artifacts they attest.

### RT-12: Checksum marker write failure was not fail closed

Severity: P2

The atomic checksum-marker writer return code was not explicitly checked before
the pipeline advanced.

Evidence:

- `scripts/qtail_orico_full_pipeline.sh:484`
- `scripts/qtail_orico_full_pipeline.sh:563`

Disposition: check the writer exit code, require the marker to exist and verify,
and bind it into the training marker.

### RT-13: Final QA had a fixed-point completion dependency

Severity: P1

The page verifier required a rendered 9/9 state before writing the final
marker, while the status builder correctly refused to publish 9/9 without a
valid final marker. A plain preview could not safely satisfy either side.

Disposition: use a PID- and expiry-bound bootstrap marker that binds immutable
training outputs and is valid only while the parent pipeline is alive. Generate
the final page, timeline, screenshots, and QA evidence under that sealing
lease; then replace the bootstrap with the full immutable marker only after the
terminal process logs are captured. Failure or owner exit removes its effect
and returns public status to 8/9.

### RT-14: Incremental closure self-test raced the live downloader

Severity: P2

The post-snapshot deferral control read mutable ledger, cache-manifest, and
record-audit files independently and expected an absolute deferred count of
one. A legitimate TFRecord completing between the prewarm snapshot and the
control could make the baseline nonzero, causing a false self-test failure
after the production closure had correctly deferred that shard.

Disposition: copy the exact bytes of all four input documents into one
temporary frozen test set before any control runs, record their snapshot
SHA-256 values, and require the synthetic post-snapshot mutation to increase
the frozen baseline by exactly one. Three consecutive controls against the
active downloader and the next production prewarm pass must all report exact
7/7, including the formal-mode rejection control.

### RT-15: Scientific direction was incorrectly coupled to completion

Severity: P0

The earlier completion gate required a positive Q-Tail outcome. That makes an
honest negative or inconclusive formal run impossible to finish and creates a
direct incentive to select favorable results.

Disposition: separate `experiment_execution_valid` and
`formal_results_publishable` from `hypothesis_outcome`. The preregistered
outcome is one of `supported`, `not_supported`, or `inconclusive`; all three
can satisfy completion when the execution and evidence gates pass.
`droid_protocol_selftest.json` now exercises all three outcomes exactly and
also rejects completion, marker, or page-verifier source contracts that
hard-code `supported=true` as a success requirement.

### RT-16: Shard arm-swap fraction was not a valid p value

Severity: P0

Source and Q-Tail allocation weights are globally coupled through softmax
normalization. Swapping one held-out shard pair changes the normalization of
all units, so the assumed independent exchangeability required by the earlier
paired randomization p value does not hold.

Disposition: publish `paired_shard_arm_swap_diagnostic_v2` only as descriptive
sensitivity evidence. Record `exchangeability=false`,
`conditional_p_value_is_valid_p_value=false`, and exclude it from support and
completion gates.

### RT-17: Formal training could be rebound to stale mirror evidence

Severity: P0

A previously valid download marker could be reused after the current data
tree, checksum ledger, or transport evidence changed.

Disposition: immediately before formal training, rebuild the complete
4,102-object binding from current files and require exact agreement with the
immutable marker, checksum ledger, transport status, 4,096 TFRecords,
3,700,745,265,151 bytes, and 187,891 records. The stage-marker verifier repeats
that current-data binding independently.

### RT-18: Checkpoint existence did not prove checkpoint semantics

Severity: P1

Hashes and filenames alone could not prove that every checkpoint contained the
expected optimizer update, environment fingerprint, training signature, model
state, and optimizer state.

Disposition: require the exact four-stage by five-step grid at steps 0, 5,000,
10,000, 15,000, and 20,000. The final verifier independently deserializes all
20 PyTorch payloads and rejects any semantic mismatch.

### RT-19: Final QA projected 9/9 before evidence was sealed

Severity: P0

The bootstrap path could make the page appear complete before the final
screenshots, timeline verification, process logs, and marker binding had been
committed.

Disposition: bootstrap and precommit QA remain visibly `in_progress` at
`8/9`, with only `final_page_qa` false. The pipeline commits the fully bound
marker first, refreshes the public projection, freezes byte-identical live and
final JSON copies under `DROID_PUBLIC_PROJECTION_COMMITTED`, and only then
accepts the published `9/9` as complete.

### RT-20: Missing artifact links looked authoritative

Severity: P1

Links to future-stage outputs were visually indistinguishable from available
evidence and could return 404 when clicked during the live run.

Disposition: derive availability from the page's complete artifact-link set.
Every link displays `READY` with byte count or `WAIT` with a stage explanation.
Desktop and mobile smoke QA require both states during the download phase and
reject console, page, resource, or overflow errors.

### RT-21: Holdout isolation is weaker than task-level generalization

Severity: P1

Release-stratified TFRecord shard holdout does not guarantee independence by
task, scene, collector, or instruction fingerprint.

Disposition: retain this as an explicit claim boundary. The formal result is
an allocation-head mechanism evaluation on held-out shards, not robot-policy
rollout success or task-isolated causal evidence.

### RT-22: Local hash chains are not external trust anchors

Severity: P1

A party with write access to the machine can rewrite both local artifacts and
their local manifests. Internal SHA-256 consistency alone does not prove when
the evidence existed or that it was never rewritten.

Disposition: label the local chain only as integrity evidence. A scoped Git
commit/PR or independent timestamp/WORM receipt remains a separate publication
step; the page must not imply that this external anchor already exists.

### RT-23: Mock-only marker tests could miss a broken terminal transition

Severity: P1

Isolated positive and tamper controls can pass while the actual 8/9 to 9/9
sequence is misordered, because replacing the final and public-state validators
also replaces the behavior the test is intended to prove.

Disposition: add eight temporary-job integration controls that call the real
lease, precommit, final-public-state, dual-marker, live/snapshot equality, and
complete final-state validators. Only the expensive upstream mirror, model,
manifest, continuity, and process-log ownership boundaries are substituted.
The suite must pass all 24 controls before formal training or final
publication.

### RT-24: Final process-log gate depended on a future event

Severity: P0

The first final seal required `qa_commit_complete`, but the parent pipeline
could write that event only after the seal and public projection had already
completed. A stale token from an older generation could pass while the current
generation could never satisfy the ordering honestly.

Disposition: bind the process-log gate to
`QTAIL_TERMINAL qa_sealing_complete qa_sha256=<sha256(final_page_qa.json)>`.
Write and hash that event before freezing the process logs. Keep
`qa_commit_complete` strictly after the dual-marker publication and verify the
entire order with positive and reversed-order shell controls.

### RT-25: Final marker commit did not enforce the bootstrap lease

Severity: P1

A caller could invoke the final-marker commit path without proving that a live
parent pipeline owned a non-expired bootstrap lease.

Disposition: `commit_final_marker` now calls the production bootstrap
validator first. Dedicated negative controls reject a missing bootstrap,
expired lease, and owner-PID mismatch before an honest 8/9 commit is accepted.

### RT-26: Final marker alone exposed a 9/9 crash window

Severity: P1

Publishing live 9/9 state before the second public-projection marker meant a
crash could leave the page looking complete without a fully sealed projection.

Disposition: hold the progress lock for the whole transition, construct the
candidate 9/9 state privately, write frozen snapshots and the second marker,
then publish the live audit followed by `latest.json` last. A valid final
marker without the second marker remains sealed but visibly 8/9.

### RT-27: Page QA could mutate an already sealed run

Severity: P2

The non-smoke verifier could delete or replace a valid final marker and could
run outside the parent pipeline that owned the sealing transaction.

Disposition: an already-complete state is now read-only. An incomplete
non-smoke run requires `qtail_orico_full_pipeline.sh` as its parent and refuses
an invalid existing final marker or an orphaned public marker.

### RT-28: Checkpoint grid inferred evidence from training text

Severity: P2

The page could mark a checkpoint `SAVED` or `VERIFIED` from a reported model
step without proving that the expected file and manifest binding existed.

Disposition: `SAVED` requires the exact nonempty checkpoint file.
`VERIFIED` additionally requires path, bytes, SHA-256, step, update count,
stage, and target agreement plus a valid final training marker. Download-phase
smoke QA requires all 20 cells to remain `WAIT`.

### RT-29: Artifact READY meant only path existence

Severity: P3

A failed QA JSON or unrelated HTTP 200 response could receive a green `READY`
badge despite contradicting the evidence claim.

Disposition: availability now requires nonempty content and stage-specific
semantic validity. Final QA JSON and screenshots stay `WAIT` until
`status=complete`; failed or malformed JSON and explicit false validity fields
also stay `WAIT`. Four temporary-file controls and live smoke QA assert these
behaviors.

### RT-30: Download-generation handoff was not self-healing

Severity: P1

The handoff watcher was durable while its process remained alive, but a reboot
or watcher loss during a multi-day download could leave a restarted pipeline
without the required one-to-one handoff binding.

Disposition: while the immutable download marker is absent, the watchdog
requires one handoff command bound to the current pipeline PID and recreates it
when missing. Once `DROID_DOWNLOAD_COMPLETE` exists, this auto-launch is
disabled so checksum and training cannot enter a restart loop. The live
runtime contract independently requires exactly one correctly bound handoff
during download and zero afterward. The scheduled launcher separately refuses
to resume before ORICO is mounted, restores progress/prewarm/watchdog services,
and binds its handoff to the current pipeline PID. Three combined controlled
source tests accept both guarded implementations and reject variants with the
mount guard, a required resume call, the download marker guard, or PID binding
removed. The independent transport LaunchAgent also keeps UniClashCore online
while rejecting DROID sockets that do not route through `en1`.

### RT-31: Equal-compute evidence relied on declared booleans

Severity: P1

The trainer reset the same seed and used the same feature matrix for each
Source/Q-Tail pair, but the sealed checkpoint evidence did not independently
prove equal initialization or equal paired inputs.

Disposition: every checkpoint now carries a content-derived feature
fingerprint and initialized-model-state fingerprint. The manifest requires
equal evaluation-pair features, equal deployment-pair features, and one common
initial state across all four stages. Step-0 state fingerprints are recomputed
from tensor content. The protocol self-test mutates both fingerprints and
requires fail-closed rejection. The pipeline, stage-marker verifier, and final
browser verifier all require the new equality gates.

### RT-32: Final mirror verification lacked its own destructive controls

Severity: P1

The full-mirror verifier consumed the official manifests, MD5 ledger, file
identity, and `gsutil -c` return code, but its own fail-closed behavior was not
sealed as a required formal artifact. It also inferred the formal object and
TFRecord counts from surrounding gates instead of enforcing them itself.

Disposition: the verifier now independently locks 4,102 objects and 4,096
TFRecords, checks the official source and object generation binding, and emits
the expected counts in its report. Eight synthetic positive/negative controls
cover checksum failure, missing and extra files, same-size content drift with
restored `mtime`, generation drift, duplicate manifest paths, and TFRecord
count drift. The self-test is a required artifact in the pipeline, environment
manifest, stage-marker validator, completion audit, and final browser QA.

During the same audit, the final browser verifier was found to expect the
obsolete runtime-process control count of 8/8 while the current authoritative
artifact and stage validator require 14/14. The browser gate now requires
14/14, preventing a false final rejection after a valid training run.

### RT-33: Training-start evidence could precede the last formal gate

Severity: P1

The trainer performed mirror and exact record closure before optimization, but
the `DROID_MODEL_TRAINING_STARTED` marker was written before the deterministic
holdout-count gate. The implementation order was also verified only by manual
source review, so a later refactor could silently move an optimizer-backed
stage ahead of a formal input gate.

Disposition: the marker now follows the exact per-release holdout check.
`droid_training_gate_order_selftest.json` runs one positive and seven
destructive source-order controls. It requires the immutable protocol, verified
mirror, all-record extraction, exact 187,891-record closure, 100% parse/scan
coverage, and release-stratified holdout gate to precede both the training
marker and first training stage; it separately rejects an optimizer update
placed before the terminal step guard. The pipeline, artifact manifest,
stage-marker validator, completion audit, page, and final browser QA all
require the 8/8 artifact.

### RT-34: Transient prewarm resets could be mistaken for data loss

Severity: P1

The original repeated prewarm loop exposed its current-pass processed-shard
counter. A new pass legitimately starts at zero, but displaying that decrease
beside persistent checksum counts made a cache-scan restart look like lost
training data.

Disposition: feature progress now publishes
`monotonic_committed_prewarm_snapshot_v1` counters from the last atomically
committed pass and keeps the current pass in a separate transient object. The
timeline verifier reports legacy `feature_pass_reset_events` separately from
`committed_feature_counter_decrease_events`; only the latter is a persistence
failure and final QA requires it to remain zero. The page exposes an
independent data-continuity card that labels legacy pass restarts as non-loss
while requiring zero official-ledger and committed-cache decreases on both
desktop and mobile.

### RT-35: The outer shell could silently relax the formal trainer invocation

Severity: P1

The trainer enforced its own formal constants, but the pipeline shell control
previously proved only that a training label existed. A later edit could remove
the verified-mirror flag, switch out of full-record mode, or reorder evidence
closure without that shell-level control identifying the exact regression.

Disposition: the seven-control shell suite now binds every formal trainer
argument and the monotonic checksum, environment, prewarm-exit, training,
full-cache recomputation, record-closure, marker, QA, log-sealing, and public
projection order. Destructive variants with the formal launch,
`--require-verified-mirror`, public commit, parent ownership, or read-only
completed-state contract removed are rejected.

### RT-36: Direct mode did not physically bind the download interface

Severity: P0

`--noproxy '*'`, disabled TUN, bypass rules, and observed `en1` routes were
strong runtime evidence, but a later route change could still move a newly
opened transfer away from the physical Wi-Fi interface.

Disposition: every official DROID curl process now carries both
`--noproxy '*'` and `--interface en1`; the downloader also fail-closes its route
preflight unless the endpoint route is exactly `en1`. Transport guard v6
requires the same command binding and independently checks every live remote
socket route. The v4 UniClashCore restart pause and v5 interface-migration pause
are retained as hashed raw epochs: both record zero forbidden proxy sockets and
zero wrong-route observations, while the current v6 epoch starts only after
all workers use the hard binding. The adjudicator and timeline verifier report
those conservative policy pauses separately from VPN-route violations.

### RT-37: Detached screen ownership broke the generation handoff

Severity: P1

The handoff tried to prove ownership by requiring the pipeline's immediate
parent to be the watchdog. A detached `screen/login` wrapper legitimately
reparents the pipeline, so the handoff exited even though the unique watchdog
and pipeline were both healthy.

Disposition: the handoff now requires exactly one exact-command watchdog and
exactly one exact-command pipeline, and requires the supplied target PID to
match that sole pipeline. The runtime contract independently requires one
handoff bound to the live pipeline during download. The repaired watcher is
kept in its own detached screen session and does not restart or interrupt the
active downloader.

### RT-37b: Marker existence alone could trigger a generation handoff

Severity: P3

The watcher previously treated the existence of `DROID_DOWNLOAD_COMPLETE` as
the transition signal. The new pipeline would reject a corrupt marker before
training, but the old generation could still be restarted unnecessarily.

Disposition: before sending `STOP`, `TERM`, or `CONT`, the handoff now reruns
the immutable download-marker verifier against the official object/checksum
manifests, complete MD5 ledger, transport status, exact byte total, and live
mirror. An invalid marker leaves the current download generation untouched.

### RT-38: Preview and final validators disagreed on control counts

Severity: P1

The live page had already advanced the downloader control display to 13/13,
while the final stage and browser validators still required the obsolete
14/14 count. Smoke mode did not execute every formal-only assertion, so the
page could look healthy throughout download and then be rejected after a
valid full training run.

Disposition: progress validation, stage-marker validation, formal browser
validation, and the visible audit text now all require 13/13 downloader
controls. Runtime controls were later expanded to 14/14 to cover the repaired
detached-screen-safe handoff ownership contract, generation-marker binding,
and their destructive mutations. The current protocol, environment,
downloader, runtime, stage sealing, and desktop/mobile smoke suites were rerun
from the same source generation before the ORICO code snapshot was resealed.

### RT-39: The support gate did not require the full minimum effect

Severity: P0

The point estimate had to reach +2 pp, but the confidence-interval lower bound
only had to exceed zero. That could label a statistically positive yet
sub-minimum effect as `supported`.

Disposition: gate v4 now requires both the point estimate and stratified 95% CI
lower bound to be at least +2 pp, plus a strictly positive extreme-tail
underallocation reduction. `not_supported` is fail-closed when the CI upper
bound is below +2 pp or the extreme reduction is nonpositive. Exact boundary
controls are part of the 38/38 protocol suite and every outer validator
recomputes the same three-state result.

### RT-40: Holdout identity depended on local path presentation

Severity: P1

A split keyed from a local or incompletely scoped filename could drift after a
mount change, path rewrite, or same-basename collision.

Disposition: holdout v2 hashes the official `release/filename` relative path,
stores all 820 sorted members, and locks their set to SHA-256
`16781c97f05cc2bdc94837b0ae96942ac9621174d60775d2c6185dae5fd8a767`.
The trainer, shell, stage marker, progress builder, browser verifier, and
protocol self-test all require the exact digest and official path scope.

### RT-41: Checkpoint file hashes did not prove tensor-tree integrity

Severity: P0

A regenerated manifest could faithfully hash a checkpoint whose model tensor
or optimizer moment had already been modified, while retaining plausible
metadata.

Disposition: checkpoint format v6 recomputes canonical model-state and
optimizer-state content hashes and links each stage checkpoint to its parent
file SHA-256. Resume, manifest construction, stage sealing, and browser QA
require the full chain. Destructive controls mutate a model tensor and an
optimizer moment independently and require rejection.

### RT-42: Bounded runs could publish a formal-looking completion sentinel

Severity: P0

An engineering smoke invocation could previously be mistaken for the formal
all-record run if it wrote the same terminal model marker.

Disposition: completion-marker publication now requires the trainer's locked
formal mode; bounded and test runs cannot create
`DROID_MODEL_TRAINING_COMPLETE`. The 38/38 protocol suite exercises the CLI
boundary directly.

### RT-43: An empty rare-fingerprint set could abort or misstate evidence

Severity: P1

The auxiliary coverage calculation assumed at least one eligible held-out
fingerprint. A legitimate empty set could either abort an otherwise valid
formal run or tempt a fabricated zero-gain curve.

Disposition: the metric now reports `no_eligible_fingerprints`, zero counts,
empty curve/time arrays, and an explicit reason. It remains auxiliary and
descriptive. The page renders `N/A`, and every validator accepts only this
exact empty shape or the complete seven-budget shape.

### RT-44: Formal artifact paths could escape the sealed root

Severity: P0

A manifest entry using `..` or a symlink could bind an artifact outside the
formal result root while appearing to have a valid hash.

Disposition: the authoritative merger now requires every formal artifact to be
an ordinary non-symlink file whose resolved path remains under the formal
DROID result root. Six controls reject parent traversal, escaped symlinks,
missing files, membership drift, and declared-path drift.

### RT-45: Resealing could mutate the manifest already bound by training

Severity: P1

Late-generated controls changed the main artifact manifest after the training
marker was committed, creating a circular mutation risk between the marker and
final evidence.

Disposition: the pipeline atomically snapshots
`droid_training_artifact_manifest.json` for the immutable training boundary,
then produces the final expanded manifest separately. Training and final
markers bind the appropriate manifest generation independently.

### RT-46: A 9/9 projection lacked a browser proof of the committed state

Severity: P1

Precommit browser QA could prove the honest 8/9 sealing page, and the public
projection could later show 9/9, but no browser run proved that the committed
9/9 page itself rendered without errors on both desktop and mobile.

Disposition: finalization now commits the 9/9 projection, runs a read-only
desktop/mobile browser pass against that exact state, stores JSON plus both
screenshots, and commits `DROID_POSTCOMMIT_PAGE_QA_COMPLETE`. Full completion
requires the final marker, public-projection marker, and postcommit browser
marker together.

### RT-46b: Concurrent postcommit QA could reseal one completion twice

Severity: P2

Two parent-owned browser QA processes could both pass the initial
marker-absence check, overwrite the same screenshot/report paths, and then
commit the postcommit marker in sequence. The final evidence could remain
valid, but the history would not prove one browser run and one immutable commit.

Disposition: postcommit browser QA now acquires a PID-bound exclusive run lock
before Chrome starts. The marker helper separately holds `flock` across
existing-marker validation, artifact validation, and commit. A valid existing
marker is returned without rewriting it; an invalid existing marker is rejected
instead of silently resealed.

### RT-47: Detached sidecars could be alive but no longer doing work

Severity: P0

The progress, watchdog, and feature-prewarm screen sessions could retain a zsh
process and a sleep child while their activity files stopped advancing. Process
counts alone therefore reported a healthy topology even though the page,
watchdog status, or record-cache snapshot was stale. A first liveness repair
also used the zsh-special local name `path`, which shadowed the command-search
array and was caught by a live no-restart probe before the next scheduled run.

Disposition: the five-minute launcher now requires both one exact-command
process and fresh activity evidence. Progress binds to `latest.json`, watchdog
binds to its atomic status file, and prewarm must either match the complete
TFRecord count or expose a fresh active extraction status. Stale or duplicate
instances are terminated with their descendants before one detached
replacement is created. Runtime contract v6 keeps 14/14 controls while adding
negative source mutations for missing freshness, missing tree cleanup, and zsh
`$path` shadowing. A live probe proved stable progress, watchdog, prewarm,
pipeline, downloader, and transport-guard PIDs across a healthy launcher pass.

The same live probe exposed a separate substring-classification false positive:
a diagnostic shell that mentioned the guard script path was counted as a second
guard. Process discovery now requires the Python process's actual script
argument to equal the guard or downloader path. The historical sample remains
in the hash chain with `runtime healthy=false`; timeline verification reports
one runtime process anomaly separately from zero VPN-route violations.

### RT-48: Clean sampled routes could be mistaken for whole-history proof

Severity: P0

The first retained download sample predates the first preserved transport-guard
sample, and one archived classifier epoch has an explicit coverage gap.
Reporting only `clean/total` full-pipeline samples could therefore be read as a
claim that the entire historical byte stream was observed.

Disposition: timeline verification now emits a separate evidence-scope object
with the pre-guard duration and filesystem-byte change, hash-preserved epoch
count, classifier-gap count, and inter-epoch gaps. The page exposes the
pre-guard evidence gap next to the zero observed VPN-route violations. Neither
the verifier nor the page treats unobserved bytes as proof of direct or VPN
transport.

### RT-49: Completion projection inverted two authoritative control counts

Severity: P1

The downloader single-writer self-test passed 13/13 and the runtime-process
contract passed 14/14, but the completion projection required 14 downloader
checks and 13 runtime checks. Both direct artifacts were green while
`intermediate_artifacts` could never become true, so a valid formal run would
have remained stuck below 9/9.

Disposition: completion now requires exactly 13 downloader controls and 14
runtime controls. Live desktop/mobile smoke QA reads all nine self-test-backed
completion fields, requires each direct artifact to report `status=passed`,
requires the corresponding projection to be true, and records the complete
artifact-to-field mapping in `live_page_smoke.json`. A regression in either
the direct evidence or its public projection therefore fails before formal
publication.

### RT-50: Frozen preflight silently lost its per-shard record cap

Severity: P1

The full-run safety override keyed only on `max_shards == 0`. A bounded run
using the newer `--shard-list` interface also leaves `max_shards` at zero, so
the trainer silently replaced `--records-per-shard 2` with all-record mode.
That made the declared bounded preflight scope disagree with the actual scan
behavior and left the published engineering evidence tied to an older trainer
generation.

Disposition: all-record enforcement now requires both `max_shards == 0` and
the absence of a frozen shard list. Two executable controls prove that a
frozen list preserves its record cap and that a genuinely unbounded mirror
scan still forces all-record mode, bringing the shard-list suite to 11/11.
The current-generation preflight independently rehashed four official
TFRecords per release against the checksum ledger, decoded exactly 16 records,
ran Source and Q-Tail for 50 MPS updates per arm, and then resumed all four
stages from step 25. Sixteen checkpoint-v6 files retained identical hashes
across terminal resume, their parent chains passed, and the formal marker set
remained unchanged. The previous public preflight summary/report are retained
under `preflight_history`; the replacement remains engineering-only and cannot
satisfy a formal effect or completion gate.

### RT-51: Post-download handoff convergence was only implicit

Severity: P1

The handoff waits for the old feature-prewarm shell to exit, while that shell
normally exits only after `DROID_CHECKSUM_VERIFIED` exists. Although the
current watchdog is restarted before the wait and can launch the new pipeline
that commits the marker, the executable control suite checked ownership and
marker binding without proving this complete ordering. A future source edit
could therefore create a circular wait while the individual process-count
checks still passed.

Disposition: runtime contract v9 expands to 16/16. It requires the handoff to
start the current watchdog before its prewarm wait, the watchdog to launch a
missing pipeline, the pipeline to commit checksum verification before its
formal prewarm wait, and prewarm to exit on that marker. Independent negative
mutations remove each edge and must all fail. The progress projection, stage
marker verifier, browser QA, and page now require the exact v9 identity and
16/16 result; stale v8 evidence cannot be promoted.

### RT-52: Any syntactically valid backend commit could pass the environment gate

Severity: P0

The environment manifest previously required only a 40-character Git commit
and a successful `git fsck`. A different commit, a forked origin, or local
tracked/untracked edits could therefore enter formal training while still
appearing reproducible.

Disposition: the formal seal now requires the official DROID policy-learning
origin, pinned commit
`9a29c832b4c81bf38401111f5e4cdddaca217581`, a completely clean worktree, and
Git object integrity. Environment contract v2 expands from 5/5 to 8/8 with
isolated negative controls for commit drift, origin drift, and an untracked
file. The progress projection and browser QA require the exact v2 identity and
control set, so older environment evidence cannot satisfy completion.

### RT-53: The final-stage closure judge lagged behind its seven-case self-test

Severity: P1

The incremental-closure self-test added a formal-mode rejection proving that
4,102 objects, 4,096 TFRecords, and 187,891 decoded records must all close
before formal training. The final-stage judge still expected the earlier
six-case shape. A valid completed mirror would therefore have produced a
seven-case artifact that the stale judge rejected after the download had
already finished.

Disposition: the judge now validates one exact seven-name contract, all seven
true checks, an empty failed-check list, the exact seven case identities, and
the positive, destructive-rejection, formal-gate-rejection, and post-snapshot
deferral semantics. Marker hardening expands to 38/38 with five independent
controls: the canonical payload passes, while a missing formal check, an extra
check, a false check, or a formal rejection spoofed as success must fail. The
live seven-case artifact passes the same helper used by the final-stage judge.

### RT-54: The page linked only 41 of 64 formal artifact paths

Severity: P1

The artifact section exposed many process logs and reports, but its static link
set omitted the 20 individual formal checkpoints and three transition gates.
The separate checkpoint matrix showed logical states without providing the
contract paths, so a reader could not inspect every required intermediate
artifact from the page. Counting all visible links concealed this 41/64 formal
contract coverage gap.

Disposition: the authoritative progress projection now publishes the sorted
`required_artifacts` set independently of the missing and unsealed subsets.
The page renders all current formal requirements in a dedicated ledger with
`SEALED`, `GENERATED`, and `WAIT` states. Browser QA requires row/count and
state-count equality, unique relative links, exactly 20 checkpoint paths, all
three post-download transition gates, and no desktop or 390-pixel mobile
overflow. Every generated or sealed contract URL must return HTTP 200, while
future-stage `WAIT` paths are not probed prematurely. The live audit currently
closes 64/64 paths as 0 sealed, 27 generated but unsealed, and 37 waiting;
formal results remain withheld.

### RT-55: A caught-up prewarm pass could look like full-mirror completion

Severity: P1

The feature prewarmer previously emitted `prewarm_complete` after it decoded
every shard available at its snapshot. During download, that label could be
read as all 4,096 official TFRecords being present even when the count was
still lower.

Disposition: prewarm status now encodes both snapshot scope and the official
4,096-shard target and the exact 2,048 + 2,048 official release composition.
Partial successful snapshots are
`prewarm_caught_up_current_snapshot`; only exact official shard coverage can be
`prewarm_full_official_shard_snapshot_complete`. A dedicated six-case
contract selftest covers partial, exact, coverage-error, excess, empty, and
wrong-release-composition inputs. Independent checksum and record gates still
control formal training. Final browser QA semantically validates the exact
six-name artifact and includes it in the final hash manifest; a missing,
renamed, extra, or false control blocks final publication.

### RT-56: Environment capture did not bind optimizer startup to one code generation

Severity: P0

The environment manifest recorded workspace SHA-256 values, but it did not
prove those values matched the atomic ORICO orchestration snapshot. The
pipeline-generation gate bound only the shell script, and the trainer did not
re-hash all critical code before entering the verified-mirror and optimizer
path. A post-capture edit could therefore leave a valid-looking environment
artifact while formal training executed a different code generation.

Disposition: environment contract v3 now requires exact workspace-to-ORICO
snapshot parity and adds a ninth destructive control for snapshot drift. The
trainer requires that environment manifest in formal mode and re-hashes every
listed critical file, the trainer itself, and the retained snapshot manifest
before mirror parsing or any optimizer update. Gate-order controls expand to
11/11 with explicit ordering, positive binding, and live-code mutation
rejection. Training and final-page judges independently recompute the same
binding through publication. Shell contract v8 adds a tenth destructive
control proving the formal trainer cannot lose its environment-manifest
argument. These local hashes establish internal retained
artifact consistency only; they are not an external timestamp or WORM seal.

### RT-57: A mount-visible LaunchAgent could kill a worker it could not replace

Severity: P0

macOS allowed the scheduled LaunchAgent to see the ORICO mount and inspect its
status files while denying writes to the removable volume. At
`2026-07-31T10:38:50Z` that launcher classified prewarm PID `81824` as stale,
terminated it, and reported a replacement two seconds later. The replacement
could not publish its first ORICO heartbeat and exited immediately. Download
and transport isolation continued, but runtime health correctly fell because
the unique prewarm process was absent. A mount check alone was therefore not a
sufficient authority check for destructive supervision.

Disposition: the terminal launcher now performs an atomic ORICO write/remove
probe before starting web supervision or inspecting, stopping, or replacing
any worker. Failure logs locally and exits without touching existing
processes. Runtime contract v10 retains the same 16 controlled cases but
extends its launcher source acceptance and destructive mutation suite to
require this guard before every supervision action. Prewarm was restored under
the ORICO-capable execution domain as PID `95778`. At the next real scheduled
run (`2026-07-31T10:43:52Z`) the LaunchAgent logged that ORICO write access was
unavailable, refused supervision, and left PID `95778` unchanged. Its bound
heartbeat then advanced through `3,782` and `3,784` complete-shard snapshots;
runtime health and browser smoke QA returned to passed. The audit proves this
observed permission boundary and source ordering, not that future macOS
privacy settings or external-volume availability cannot change.

### RT-58: Training completion could expose values before final publication

Severity: P0

The live page treated a valid `DROID_TRAINING_COMPLETE` marker as sufficient
to render formal values. That marker proves the same-compute experiment ran,
but it precedes the independent final artifact seal, desktop/mobile QA, and
atomic public projection commit. A short 8/9 interval could therefore expose
an unsealed result as if it were already public evidence.

Disposition: `formal_results_publishable` now remains false until the final
page marker and `DROID_PUBLIC_PROJECTION_COMMITTED` both validate. The page
requires the same three conditions and stays `WITHHELD` throughout the 8/9
precommit QA. Actual formal values are rendered only from the committed 9/9
projection and are then checked by the read-only postcommit desktop/mobile
QA. Marker hardening adds two destructive controls: honest precommit
withholding must pass, while a training-only publication flag must be
rejected. A separate identity-contract control locks the exact 38-name set and
rejects missing, extra, or duplicate controls. This expands the marker suite
to 38/38 without changing the
hypothesis outcome or making outcome direction a completion gate.

### RT-59: The formal rejection control collapsed after the mirror became full

Severity: P0

The incremental-closure self-test derived its formal-mode expectation from the
live baseline. Before download completion that correctly exercised rejection,
but after all 4,102 objects, 4,096 TFRecords, and 187,891 records closed, the
same case became a successful full-mirror invocation. The final judge correctly
required a rejection control and withheld the training marker, so a complete
training run could not be published even though its model artifacts were
valid.

Disposition: the formal rejection case now reuses the frozen post-snapshot
fixture with exactly one TFRecord deferred. Ordinary incremental closure must
still return zero, while the same fixture under `--require-formal` must return
nonzero with `formal_full_mirror_gate=false`. The live full-mirror positive
case remains separate. The corrected seven-case artifact passes the exact
final-judge helper, and the failed publication attempt is retained under
`run_attempts/` rather than overwritten.

### RT-60: The page was healthy while an immutable marker URL returned 404

Severity: P1

Both supervised `serve` instances returned the expected DROID page and passed
the runtime content probe, but `download_completion_marker.json` returned 404.
The result entry is intentionally a symbolic link to the immutable marker in
the same ORICO job root, and `serve` rejects symbolic links by default. A judge
could therefore see a healthy page and a READY artifact label without being
able to retrieve the evidence.

Disposition: both supervised endpoints now use the explicit
`serve --symlinks` mode and bind ownership checks to that exact command.
Runtime contract v11 retains 16 named controls while strengthening the web
source control: removing `--symlinks` must make the destructive suite fail.
Desktop/mobile smoke QA additionally requests every READY artifact and requires
HTTP 200. The observed download marker now returns 200 on ports 54655 and 6222.
This is a local evidence-retrieval guarantee for the audited job root, not a
public hosting, authorization, or external immutability claim.

### RT-61: Same-machine checkpoints could cross formal code generations

Severity: P0

The formal trainer originally bound resumable checkpoints only to hardware,
OS, Python, PyTorch, MPS, and device. A completed checkpoint from an earlier
formal code snapshot therefore remained eligible on the same machine even
after the environment manifest and atomic ORICO orchestration snapshot had
changed. The startup environment audit was current, but zero optimizer updates
could be inherited from the current generation.

Disposition: checkpoint format v6 introduces
`qtail_checkpoint_environment_v2`. Its fingerprint covers the runtime
environment plus the formal environment-manifest SHA-256, checked live-code
aggregate SHA-256, ORICO snapshot-manifest SHA-256, and snapshot code-parity
result. Resume, intermediate-manifest, final-marker, progress, preflight, and
browser validators all compare this dedicated fingerprint. A deterministic
negative control holds the runtime environment fixed, changes only the formal
snapshot hash, and requires rejection with restart from step 0. Earlier v5
checkpoints remain retained as failed-generation evidence but cannot resume a
v6 run.

## Acceptance rule

No red-team remediation may interrupt the live download generation. Updated
code becomes active at the existing immutable download-completion handoff. The
formal training and final page remain withheld until the remediated controls,
full mirror closure, full record scan, and supported, not-supported, or
inconclusive result reporting all pass their independent validators.
