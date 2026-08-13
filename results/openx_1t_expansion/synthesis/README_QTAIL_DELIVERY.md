# Q-Tail PT-Heavy-Tail Synthetic Data Delivery

## What This Package Is

This is a Q-Tail-for-AI delivery package for embodied-AI data teams. It takes a customer task/trajectory summary CSV, scores rare and risky tasks, then emits a PT-heavy-tail synthetic allocation plan plus a same-budget audit.

The current Open X stage trains a record-informed allocation head on real downloaded Open X / RT-X RLDS TFRecord shards. Every complete shard is covered with bounded episode decoding. The final Strong run is gated until the selected add-on datasets finish downloading and pass completeness checks.

## Current Evidence Summary

- Winner: qtail_synthetic
- Gate passed: True
- Tail success: 35.6% -> 70.7% (+35.1 pp, relative 98.4%)
- CVaR@20: 31.1% -> 70.6% (+39.5 pp)
- Tail data share: 2.0% -> 39.0% (+37.0 pp)
- Aligned with PT-heavy-tail goal: True

## Open X Calibration Source

- Training report: `/Volumes/ORICO/qtail_full_training/results/qtail_openx_1t_expansion/training/openx_demo_training_report.json`
- Training rows: `/Volumes/ORICO/qtail_full_training/results/qtail_openx_1t_expansion/training/openx_shard_training_rows.csv`
- Status: complete
- Steps: 20000
- Downloaded data used by current snapshot: 1195.523 GiB
- Shards: 3673
- Decoded episodes: 12557
- TFRecord parse coverage: 100.0%
- Model checkpoint: `/Volumes/ORICO/qtail_full_training/results/qtail_openx_1t_expansion/training/qtail_allocation_head.pt`
- Learned tail share prior: source 19.3% -> Q-Tail 53.4%
- Predicted tail share gain from trained allocation head: +34.1 pp

## Files In This Package

- `task_profiles.csv`: normalized customer task profile with tail scores.
- `qtail_synthetic_data.csv`: base Q-Tail synthetic allocation output.
- `qtail_service_synthetic_plan.csv`: OpenX-calibrated synthetic scenario/spec plan for downstream rendering.
- `per_task_comparison.csv`: same-budget source vs Q-Tail per-task comparison.
- `qtail_data_engine_report.json`: machine-readable evaluation report.
- `qtail_service_model_card.json`: OpenX-calibrated service model card.
- `qtail_service_delivery_report.json`: delivery summary, effect metrics, claim boundary, and package paths.
- `README_QTAIL_DELIVERY.md`: this handoff note.
- `qtail_delivery_package.zip`: archive containing the full package (`/Volumes/ORICO/qtail_full_training/results/qtail_openx_1t_expansion/synthesis/qtail_delivery_package.zip`).

## How To Reproduce Locally

```bash
python3 tools/qtail_openx_service_model.py \
  --input data/embodied_public_anchor_real.csv \
  --out results/qtail_openx_service_public \
  --training-report results/openx_incremental_training_snapshot/openx_demo_training_report.json \
  --training-rows results/openx_incremental_training_snapshot/openx_shard_training_rows.csv \
  --allow-inconclusive

python3 tools/qtail_validate_package.py results/qtail_openx_service_public/qtail_data_engine_report.json
```

## API Usage

```bash
curl -X POST http://127.0.0.1:8223/generate \
  -H 'Content-Type: application/json' \
  --data '{"filename":"customer.csv","csv_text":"task,count,success_rate,difficulty,group\nrare_pick,12,0.32,0.91,tail\nstandard_pick,540,0.86,0.22,head\n","synthetic_budget":100000,"top_k":128}'
```

## Claim Boundary

- The Open X stage trains a record-informed allocation head on real downloaded RLDS TFRecord shards.
- Every complete shard is covered, with a bounded number of decoded episodes per shard; this is not an all-episode policy run.
- The service package generates allocation/scenario specs for synthetic data production.
- Full robot-policy validation remains a later same-policy training run after the full RLDS/TFDS stack is ready.
- The service package validates data allocation quality and synthetic-data targeting before expensive robot-policy retraining.
- The final 20000-step Strong result will replace this incremental snapshot after download verification succeeds.

## Business Use

Q-Tail is useful when an embodied-AI team has enough common-case data but lacks coverage on rare, high-risk, or failure-prone tasks. The product value is that a customer can submit data summaries, receive a prioritized PT-heavy-tail synthetic data plan, and decide where to spend data-generation or robot-training budget before running full policy training.

Generated at: 2026-08-12T04:15:21.109644+00:00
Evaluation report: `/Volumes/ORICO/qtail_full_training/results/qtail_openx_1t_expansion/synthesis/qtail_data_engine_report.json`
