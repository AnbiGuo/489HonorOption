# Spatial Relation Localization with GroundingDINO

This project studies whether an open-vocabulary detector can localize the correct target when the text prompt includes a spatial relation, such as `the chair near the table` or `the tree in front of the building`.

## Current Scope
- Detector: GroundingDINO
- Dataset source: Visual Genome relationships
- Task: relation-conditioned localization
- Kept relation set: `above`, `below`, `behind`, `in front of`, `next to`, `near`

## What Is In This Repo
- `scripts/`: data preparation, image download, inference, and evaluation scripts
- `benchmarks/`: cleaned benchmark CSVs that are small enough to keep in Git
- `docs/`: workflow summary and current experiment notes
- `environment.yml`, `requirements_lock.txt`: local environment records

## Current Result Snapshot
The first cleaned Visual Genome run used 93 samples. Automatic evaluation at IoU 0.5 produced:
- `top1_acc@0.5 = 0.3441`
- `any_hit_acc@0.5 = 0.4731`
- `target_phrase_acc@0.5 = 0.4516`

This suggests the detector is unstable under relation prompts, but it also finds the correct target in non-top-1 positions often enough to justify a later relation-aware reranking step.

## Notes
Raw data, downloaded images, model weights, and run outputs are intentionally ignored in Git because they are large and can be regenerated locally.
