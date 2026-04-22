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

Recent experiment scripts include:
- `scripts/run_relation_baseline_suite.py`: compares full prompts, target-only prompts, and counterfactual relation prompts
- `scripts/run_geometry_reranking_experiment.py`: tests lightweight geometry-based reranking on top of detector candidates

## Current Result Snapshot
The first cleaned Visual Genome run used 93 samples. Automatic evaluation at IoU 0.5 produced:
- `top1_acc@0.5 = 0.3441`
- `any_hit_acc@0.5 = 0.4731`
- `target_phrase_acc@0.5 = 0.4516`

The follow-up relation-sensitivity analysis showed:
- `target-only top1_acc@0.5 = 0.4301`
- `counterfactual top1_acc@0.5 = 0.3441`
- full-prompt vs. counterfactual same-top1-box rate `= 0.8280`

This suggests the detector behaves much more like a noun-driven localizer than a relation-sensitive localizer.

A small follow-up reranking study on the 2D-geometry-friendly relations (`above`, `below`, `near`, `next to`) found:
- detected-reference reranking did not improve over the candidate baseline
- oracle-reference reranking produced a small gain on the subset when applied selectively to `near`

That pattern suggests explicit geometry can help, but reliable reference grounding is a major bottleneck.

## Notes
Raw data, downloaded images, model weights, and run outputs are intentionally ignored in Git because they are large and can be regenerated locally.
