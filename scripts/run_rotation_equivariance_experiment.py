import argparse
import csv
import json
import os
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers import OwlViTForObjectDetection, OwlViTProcessor


RELATION_180_MAP = {
    "above": "below",
    "below": "above",
}

IRREGULAR_SINGULARS = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "mice": "mouse",
    "geese": "goose",
    "teeth": "tooth",
    "feet": "foot",
}


def ensure_prompt_terminator(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return prompt
    return prompt if prompt.endswith(".") else f"{prompt}."


def singularize_token(token: str) -> str:
    if token in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[token]
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def canonical_name(text: str) -> str:
    tokens = re.findall(r"[a-z0-9']+", str(text).lower())
    normalized = [singularize_token(token) for token in tokens]
    return " ".join(normalized).strip()


def load_benchmark(path: Path, relations: set[str]) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["normalized_relation"] in relations]
    rows.sort(key=lambda row: row["sample_id"])
    return rows


def download_image(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, out_path)


def resolve_image_path(row: dict, images_dir: Path) -> Path:
    local_image = row.get("local_image", "")
    if local_image:
        path = Path(local_image)
        if path.exists():
            return path
    filename = os.path.basename(row["image_url"])
    path = images_dir / filename
    download_image(row["image_url"], path)
    return path


def transform_bbox_180(box: list[float], image_width: float, image_height: float) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        image_width - x2,
        image_height - y2,
        image_width - x1,
        image_height - y1,
    ]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def run_prompt(processor, model, image: Image.Image, prompt: str, device: str, box_threshold: float, text_threshold: float) -> list[dict]:
    inputs = processor(images=image, text=ensure_prompt_terminator(prompt), return_tensors="pt")
    if device != "cpu":
        inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    text_labels = results.get("text_labels")
    if text_labels is None:
        text_labels = [str(label) for label in results.get("labels", [])]

    predictions = []
    for score, box, phrase in zip(results["scores"], results["boxes"], text_labels):
        predictions.append(
            {
                "box_xyxy": [float(value) for value in box.tolist()],
                "score": float(score),
                "phrase": str(phrase),
            }
        )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions


def run_owlvit_prompt(processor, model, image: Image.Image, prompt: str, device: str, threshold: float) -> list[dict]:
    inputs = processor(text=[[prompt]], images=image, return_tensors="pt")
    if device != "cpu":
        inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device if device != "cpu" else "cpu")
    if hasattr(processor, "post_process_object_detection"):
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)[0]
    else:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            text_labels=[[prompt]],
        )[0]

    text_labels = results.get("text_labels") or [prompt] * len(results["scores"])
    predictions = []
    for score, box, phrase in zip(results["scores"], results["boxes"], text_labels):
        predictions.append(
            {
                "box_xyxy": [float(value) for value in box.tolist()],
                "score": float(score),
                "phrase": str(phrase),
            }
        )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions


def evaluate_predictions(row: dict, condition: str, prompt: str, predictions: list[dict], gt_box: list[float], iou_threshold: float) -> dict:
    target_name = canonical_name(row["target_name"])
    top1 = predictions[0] if predictions else None
    top1_iou = iou_xyxy(top1["box_xyxy"], gt_box) if top1 else 0.0
    best_iou = max((iou_xyxy(pred["box_xyxy"], gt_box) for pred in predictions), default=0.0)
    best_target_phrase_iou = max(
        (
            iou_xyxy(pred["box_xyxy"], gt_box)
            for pred in predictions
            if target_name in canonical_name(pred.get("phrase", ""))
        ),
        default=0.0,
    )
    return {
        "sample_id": row["sample_id"],
        "image_id": row["image_id"],
        "relation": row["normalized_relation"],
        "condition": condition,
        "prompt": prompt,
        "target_name": row["target_name"],
        "reference_name": row["reference_name"],
        "num_predictions": len(predictions),
        "top1_iou": round(top1_iou, 4),
        "best_iou_any": round(best_iou, 4),
        "best_iou_target_phrase": round(best_target_phrase_iou, 4),
        "top1_hit_iou50": int(top1_iou >= iou_threshold),
        "any_hit_iou50": int(best_iou >= iou_threshold),
        "target_phrase_hit_iou50": int(best_target_phrase_iou >= iou_threshold),
        "top1_score": round(top1["score"], 4) if top1 else 0.0,
        "top1_phrase": top1["phrase"] if top1 else "",
        "top1_box_xyxy": json.dumps(top1["box_xyxy"]) if top1 else "",
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"num_samples": 0}

    condition_groups = defaultdict(list)
    relation_groups = defaultdict(list)
    for row in rows:
        condition_groups[row["condition"]].append(row)
        relation_groups[(row["condition"], row["relation"])].append(row)

    by_condition = {}
    for condition, group in sorted(condition_groups.items()):
        total = len(group)
        per_relation = []
        for (group_condition, relation), rel_group in sorted(relation_groups.items()):
            if group_condition != condition:
                continue
            rel_total = len(rel_group)
            per_relation.append(
                {
                    "relation": relation,
                    "num_samples": rel_total,
                    "top1_acc_iou50": round(sum(item["top1_hit_iou50"] for item in rel_group) / rel_total, 4),
                    "any_hit_acc_iou50": round(sum(item["any_hit_iou50"] for item in rel_group) / rel_total, 4),
                }
            )
        by_condition[condition] = {
            "num_samples": total,
            "top1_acc_iou50": round(sum(item["top1_hit_iou50"] for item in group) / total, 4),
            "any_hit_acc_iou50": round(sum(item["any_hit_iou50"] for item in group) / total, 4),
            "target_phrase_acc_iou50": round(sum(item["target_phrase_hit_iou50"] for item in group) / total, 4),
            "avg_num_predictions": round(sum(item["num_predictions"] for item in group) / total, 4),
            "per_relation": per_relation,
        }
    return {
        "num_samples": len({row["sample_id"] for row in rows}),
        "conditions": by_condition,
    }


def top1_box(row: dict) -> list[float] | None:
    raw = row.get("top1_box_xyxy", "")
    if not raw:
        return None
    return json.loads(raw)


def compare_conditions(rows: list[dict], left_condition: str, right_condition: str) -> dict:
    left = {row["sample_id"]: row for row in rows if row["condition"] == left_condition}
    right = {row["sample_id"]: row for row in rows if row["condition"] == right_condition}
    sample_ids = sorted(set(left) & set(right))
    if not sample_ids:
        return {"num_samples": 0}

    same_top1 = 0
    left_only_correct = 0
    right_only_correct = 0
    both_correct = 0
    both_wrong = 0

    for sample_id in sample_ids:
        left_box = top1_box(left[sample_id])
        right_box = top1_box(right[sample_id])
        if left_box and right_box and iou_xyxy(left_box, right_box) >= 0.5:
            same_top1 += 1

        left_hit = int(left[sample_id]["top1_hit_iou50"])
        right_hit = int(right[sample_id]["top1_hit_iou50"])
        if left_hit and right_hit:
            both_correct += 1
        elif left_hit and not right_hit:
            left_only_correct += 1
        elif not left_hit and right_hit:
            right_only_correct += 1
        else:
            both_wrong += 1

    total = len(sample_ids)
    return {
        "num_samples": total,
        "same_top1_box_iou50_rate": round(same_top1 / total, 4),
        "left_only_correct": left_only_correct,
        "right_only_correct": right_only_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 180-degree rotation equivariance test for above/below relation-conditioned localization.")
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--model-family", default="groundingdino", choices=["groundingdino", "owlvit"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--owlvit-threshold", type=float, default=0.10)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir) if args.images_dir else output_dir / "images"

    rows = load_benchmark(Path(args.benchmark_csv), set(RELATION_180_MAP))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable. Falling back to CPU.")
            device = "cpu"

    print(f"Using device: {device}")
    print(f"Samples: {len(rows)}")
    if args.model_family == "groundingdino":
        processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id, local_files_only=True)
    else:
        processor = OwlViTProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
        model = OwlViTForObjectDetection.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    result_rows = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['sample_id']}")
        image_path = resolve_image_path(row, images_dir)
        original_relation = row["normalized_relation"]
        transformed_relation = RELATION_180_MAP[original_relation]
        original_prompt = row["prompt"]
        transformed_prompt = f"the {row['target_name']} {transformed_relation} the {row['reference_name']}"
        target_only_prompt = f"the {row['target_name']}"

        target_box = json.loads(row["target_bbox_xyxy"])
        image_width = float(row["image_width"])
        image_height = float(row["image_height"])
        rotated_target_box = transform_bbox_180(target_box, image_width, image_height)

        with Image.open(image_path) as image_file:
            original_image = image_file.convert("RGB")
            rotated_image = original_image.transpose(Image.Transpose.ROTATE_180)

            condition_specs = [
                ("original_full_prompt", original_image, original_prompt, target_box),
                ("rotated_consistent_prompt", rotated_image, transformed_prompt, rotated_target_box),
                ("rotated_mismatch_prompt", rotated_image, original_prompt, rotated_target_box),
                ("rotated_target_only", rotated_image, target_only_prompt, rotated_target_box),
            ]

            for condition, image, prompt, gt_box in condition_specs:
                if args.model_family == "groundingdino":
                    predictions = run_prompt(
                        processor,
                        model,
                        image,
                        prompt,
                        device,
                        args.box_threshold,
                        args.text_threshold,
                    )
                else:
                    predictions = run_owlvit_prompt(
                        processor,
                        model,
                        image,
                        prompt,
                        device,
                        args.owlvit_threshold,
                    )
                result_rows.append(
                    evaluate_predictions(
                        row,
                        condition,
                        prompt,
                        predictions,
                        gt_box,
                        args.iou_threshold,
                    )
                )

    write_csv(output_dir / "rotation_equivariance_per_sample.csv", result_rows)

    summary = summarize(result_rows)
    summary.update(
        {
            "model_id": args.model_id,
            "model_family": args.model_family,
            "device": device,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "owlvit_threshold": args.owlvit_threshold,
            "iou_threshold": args.iou_threshold,
            "relation_transform": RELATION_180_MAP,
            "comparisons": {
                "rotated_consistent_vs_mismatch": compare_conditions(result_rows, "rotated_consistent_prompt", "rotated_mismatch_prompt"),
                "rotated_consistent_vs_target_only": compare_conditions(result_rows, "rotated_consistent_prompt", "rotated_target_only"),
                "original_vs_rotated_consistent": compare_conditions(result_rows, "original_full_prompt", "rotated_consistent_prompt"),
            },
            "interpretation_note": (
                "For rotated_mismatch_prompt, a high top1 hit against the rotated original target "
                "means the model still selected the same target object even though the relation text was not transformed."
            ),
        }
    )

    with open(output_dir / "rotation_equivariance_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
