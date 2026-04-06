# Workflow Summary

## 1. Research Goal
The project asks a focused question: when a prompt contains a spatial relation, can an open-vocabulary detector return the correct target box instead of just any object from the right category?

Example prompts:
- `the bottle above the chair`
- `the chair near the table`
- `the tree in front of the building`

## 2. Environment Bring-Up
The environment was built around GroundingDINO with GPU inference. The final working setup uses:
- Python 3.10 in a local conda prefix environment
- PyTorch CUDA wheels
- GroundingDINO compiled with its CUDA extension

This phase included several dead ends, including a discarded COCO smoke-test path and multiple dependency issues. Those temporary files have been removed.

## 3. Dataset Choice
We chose Visual Genome instead of manually annotating a new dataset because it already provides:
- object boxes
- subject-object relationships
- image metadata

The raw files used locally were:
- `data/visual_genome/objects.json`
- `data/visual_genome/relationships.json`
- `data/visual_genome/image_data.json`

## 4. Benchmark Construction
The benchmark was built in stages.

1. `scripts/build_vg_relation_benchmark.py`
Maps noisy Visual Genome predicates into a smaller normalized spatial relation set.

2. `scripts/filter_vg_relation_benchmark.py`
Creates a cleaner subset by keeping conservative raw predicates only.

3. `scripts/download_vg_benchmark_images.py`
Downloads only the images referenced by the cleaned benchmark and records local paths.

4. `scripts/build_groundingdino_manifest.py`
Converts the local benchmark CSV into the 3-column manifest format required by the batch inference runner.

The final kept benchmark is:
- `benchmarks/relation_benchmark_vg_clean.csv`

Its cleaned distribution is:
- `behind`: 20
- `in front of`: 20
- `next to`: 20
- `near`: 20
- `above`: 11
- `below`: 2

## 5. Inference Pipeline
The reusable inference scripts are:
- `scripts/run_groundingdino_single.py`
- `scripts/run_groundingdino_manifest.py`

The first script is useful for single-image validation. The second runs a full manifest and saves one folder per sample with:
- an annotated image
- a JSON file of model predictions

## 6. Evaluation Pipeline
`scripts/evaluate_vg_relation_predictions.py` compares predictions against the target bounding box and reports:
- top-1 hit at IoU 0.5
- any-hit at IoU 0.5
- target-phrase hit at IoU 0.5
- error types such as `top1_correct`, `found_but_not_top1`, `wrong_localization`, and `no_prediction`

## 7. Current Results
The first cleaned Visual Genome run used 93 samples.

Overall summary:
- `top1_acc@IoU0.5 = 0.3441`
- `any_hit_acc@IoU0.5 = 0.4731`
- `target_phrase_acc@IoU0.5 = 0.4516`
- `avg_num_predictions = 1.3871`

Error breakdown:
- `top1_correct = 32`
- `found_but_not_top1 = 12`
- `wrong_localization = 40`
- `no_prediction = 9`

Per-relation behavior from the first run:
- `in front of` and `next to` were relatively stronger
- `behind` and `near` were clearly harder
- `below` is too small to draw a strong conclusion

## 8. Interpretation
The current results support the original research motivation:
- relation-conditioned localization is harder than plain category detection
- the detector is often unstable under relation prompts
- some failures are ranking failures rather than complete misses

That last point is especially important because it motivates a lightweight next step: relation-aware reranking rather than full retraining.

## 9. What Was Removed During Cleanup
The following paths were intentionally removed because they were only temporary setup artifacts:
- the public COCO mini smoke-test data and outputs
- the initial noisy Visual Genome benchmark CSV
- raw zip archives already replaced by extracted JSON files
- ad hoc test images and shell leftovers

## 10. Recommended Next Step
The clean next step is not to expand the dataset yet. It is to do one of the following:
- build an error-analysis notebook or script that surfaces representative failure cases
- add a simple relation-aware reranker on top of detector candidate boxes
