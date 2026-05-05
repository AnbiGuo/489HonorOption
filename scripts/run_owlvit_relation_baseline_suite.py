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
from transformers import OwlViTForObjectDetection, OwlViTProcessor


COUNTERFACTUAL_RELATION_MAP = {
    "above": "below",
    "below": "above",
    "behind": "in front of",
    "in front of": "behind",
    "next to": "above",
    "near": "behind",
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

VARIANTS = ("full_prompt", "target_only", "counterfactual")


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


def prompt_for_variant(row: dict, variant: str) -> str:
    if variant == "full_prompt":
        return row["prompt"]
    if variant == "target_only":
        return f"the {row['target_name']}"
    if variant == "counterfactual":
        relation = COUNTERFACTUAL_RELATION_MAP[row["normalized_relation"]]
        return f"the {row['target_name']} {relation} the {row['reference_name']}"
    raise ValueError(f"Unsupported variant: {variant}")


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def classify_error(num_predictions: int, top1_iou: float, best_iou_any: float, iou_threshold: float) -> str:
    if num_predictions == 0:
        return "no_prediction"
    if top1_iou >= iou_threshold:
        return "top1_correct"
    if best_iou_any >= iou_threshold:
        return "found_but_not_top1"
    return "wrong_localization"


def load_benchmark(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
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


def run_prompt(processor, model, image: Image.Image, prompt: str, device: str, threshold: float) -> list[dict]:
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

    predictions = []
    text_labels = results.get("text_labels") or [prompt] * len(results["scores"])
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


def evaluate_predictions(row: dict, variant: str, prompt: str, predictions: list[dict], iou_threshold: float) -> dict:
    gt_box = json.loads(row["target_bbox_xyxy"])
    target_name = canonical_name(row["target_name"])

    scored = []
    for pred in predictions:
        phrase = pred["phrase"]
        pred_box = pred["box_xyxy"]
        scored.append(
            {
                "box_xyxy": pred_box,
                "score": float(pred["score"]),
                "phrase": phrase,
                "iou_with_target": iou_xyxy(pred_box, gt_box),
                "matches_target_phrase": target_name in canonical_name(phrase),
            }
        )

    top1 = scored[0] if scored else None
    top1_iou = top1["iou_with_target"] if top1 else 0.0
    best_iou_any = max((item["iou_with_target"] for item in scored), default=0.0)
    best_iou_target_phrase = max((item["iou_with_target"] for item in scored if item["matches_target_phrase"]), default=0.0)
    error_type = classify_error(len(scored), top1_iou, best_iou_any, iou_threshold)

    return {
        "sample_id": row["sample_id"],
        "image_id": row["image_id"],
        "relation": row["normalized_relation"],
        "prompt_variant": variant,
        "prompt_used": prompt,
        "target_name": row["target_name"],
        "reference_name": row["reference_name"],
        "num_predictions": len(scored),
        "num_hit_predictions_iou50": sum(1 for item in scored if item["iou_with_target"] >= iou_threshold),
        "top1_iou": round(top1_iou, 4),
        "best_iou_any": round(best_iou_any, 4),
        "best_iou_target_phrase": round(best_iou_target_phrase, 4),
        "top1_score": round(top1["score"], 4) if top1 else 0.0,
        "top1_phrase": top1["phrase"] if top1 else "",
        "top1_box_xyxy": json.dumps(top1["box_xyxy"]) if top1 else "",
        "top1_hit_iou50": int(top1_iou >= iou_threshold),
        "any_hit_iou50": int(best_iou_any >= iou_threshold),
        "target_phrase_hit_iou50": int(best_iou_target_phrase >= iou_threshold),
        "error_type": error_type,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_variant_results(rows: list[dict], iou_threshold: float) -> dict:
    if not rows:
        return {"num_samples": 0, "iou_threshold": iou_threshold}

    relation_groups = defaultdict(list)
    error_counter = Counter()
    for row in rows:
        relation_groups[row["relation"]].append(row)
        error_counter[row["error_type"]] += 1

    per_relation = []
    for relation, group in sorted(relation_groups.items()):
        total = len(group)
        per_relation.append(
            {
                "relation": relation,
                "num_samples": total,
                "top1_acc_iou50": round(sum(item["top1_hit_iou50"] for item in group) / total, 4),
                "any_hit_acc_iou50": round(sum(item["any_hit_iou50"] for item in group) / total, 4),
                "target_phrase_acc_iou50": round(sum(item["target_phrase_hit_iou50"] for item in group) / total, 4),
                "avg_num_predictions": round(sum(item["num_predictions"] for item in group) / total, 4),
            }
        )

    total_rows = len(rows)
    return {
        "num_samples": total_rows,
        "iou_threshold": iou_threshold,
        "top1_acc_iou50": round(sum(row["top1_hit_iou50"] for row in rows) / total_rows, 4),
        "any_hit_acc_iou50": round(sum(row["any_hit_iou50"] for row in rows) / total_rows, 4),
        "target_phrase_acc_iou50": round(sum(row["target_phrase_hit_iou50"] for row in rows) / total_rows, 4),
        "avg_num_predictions": round(sum(row["num_predictions"] for row in rows) / total_rows, 4),
        "error_breakdown": dict(error_counter),
        "per_relation": per_relation,
    }


def top1_box_from_result(result: dict) -> list[float] | None:
    raw = result.get("top1_box_xyxy", "")
    if not raw:
        return None
    return json.loads(raw)


def compare_result_sets(left_rows: list[dict], right_rows: list[dict]) -> dict:
    left_by_id = {row["sample_id"]: row for row in left_rows}
    right_by_id = {row["sample_id"]: row for row in right_rows}
    sample_ids = sorted(set(left_by_id) & set(right_by_id))
    if not sample_ids:
        return {"num_samples": 0}

    same_box_count = 0
    left_only_correct = 0
    right_only_correct = 0
    both_correct = 0
    both_wrong = 0
    for sample_id in sample_ids:
        left = left_by_id[sample_id]
        right = right_by_id[sample_id]
        left_box = top1_box_from_result(left)
        right_box = top1_box_from_result(right)
        if left_box and right_box and iou_xyxy(left_box, right_box) >= 0.5:
            same_box_count += 1

        left_hit = int(left["top1_hit_iou50"])
        right_hit = int(right["top1_hit_iou50"])
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
        "same_top1_box_iou50_rate": round(same_box_count / total, 4),
        "left_only_correct": left_only_correct,
        "right_only_correct": right_only_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OWL-ViT relation-sensitivity baselines on the Visual Genome relation benchmark.")
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--model-id", default="google/owlvit-base-patch32")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir) if args.images_dir else output_dir / "images"

    rows = load_benchmark(Path(args.benchmark_csv))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable. Falling back to CPU.")
            device = "cpu"

    print(f"Loading model: {args.model_id}")
    print(f"Using device: {device}")
    processor = OwlViTProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model = OwlViTForObjectDetection.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    variant_rows = {variant: [] for variant in VARIANTS}
    total_samples = len(rows)

    for idx, row in enumerate(rows, start=1):
        image_path = resolve_image_path(row, images_dir)
        print(f"[{idx}/{total_samples}] {row['sample_id']}")
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            for variant in VARIANTS:
                prompt = prompt_for_variant(row, variant)
                predictions = run_prompt(processor, model, image, prompt, device, args.threshold)
                variant_rows[variant].append(evaluate_predictions(row, variant, prompt, predictions, args.iou_threshold))

    variant_summaries = {}
    for variant, result_rows in variant_rows.items():
        write_csv(output_dir / f"{variant}_per_sample.csv", result_rows)
        summary = summarize_variant_results(result_rows, args.iou_threshold)
        variant_summaries[variant] = summary
        with open(output_dir / f"{variant}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    suite_summary = {
        "model_id": args.model_id,
        "model_family": "owlvit",
        "device": device,
        "threshold": args.threshold,
        "iou_threshold": args.iou_threshold,
        "num_samples": total_samples,
        "counterfactual_relation_map": COUNTERFACTUAL_RELATION_MAP,
        "variants": variant_summaries,
        "comparisons": {
            "full_vs_target_only": compare_result_sets(variant_rows["full_prompt"], variant_rows["target_only"]),
            "full_vs_counterfactual": compare_result_sets(variant_rows["full_prompt"], variant_rows["counterfactual"]),
        },
    }

    with open(output_dir / "suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(suite_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
