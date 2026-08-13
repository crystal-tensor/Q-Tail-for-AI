# Q-Tail Portable Tail Synthesizer

This directory contains a trained, portable allocation model for turning a new source CSV into a long-tail synthetic allocation and an optional materialized resampled dataset.

## Train four candidates in parallel

```bash
python3 train.py \
  --data /Volumes/ORICO/qtail_full_training/results/qtail_droid_full/droid_shard_training_rows.csv \
  --out /Volumes/ORICO/qtail_tail_synthesis_model \
  --steps 6000
```

## Synthesize from a new source CSV

```bash
python3 synthesize.py \
  --model /Volumes/ORICO/qtail_tail_synthesis_model/production_model.pt \
  --source sample_source.csv \
  --out /Volumes/ORICO/qtail_tail_synthesis_model/example_output \
  --budget 10000 \
  --materialize
```

Accepted semantic columns include `task`, `task_id`, `skill`, `instruction`, `count`, `episodes`, `success_rate`, `reward`, `difficulty`, `risk`, and `tail_score`.

The model generates a learned long-tail allocation and deterministic resampled task rows. It does not fabricate raw camera frames, actions, or robot trajectories; those require a downstream domain renderer or policy-data generator.
