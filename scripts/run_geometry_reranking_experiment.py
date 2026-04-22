import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_RELATIONS = ("above", "below", "next to", "near")
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def center_xy(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_wh(box: list[float]) -> tuple[float, float]:
    return (max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1]))


def overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


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


def relation_score(target_box: list[float], reference_box: list[float], relation: str, image_width: float, image_height: float) -> float:
    tx, ty = center_xy(target_box)
    rx, ry = center_xy(reference_box)
    tw, th = box_wh(target_box)
    rw, rh = box_wh(reference_box)

    dx = abs(tx - rx) / max(image_width, 1.0)
    dy = abs(ty - ry) / max(image_height, 1.0)
    center_dist = math.sqrt((tx - rx) ** 2 + (ty - ry) ** 2)
    diag = math.sqrt(image_width ** 2 + image_height ** 2)
    center_close = 1.0 - clamp01(center_dist / max(diag * 0.45, 1.0))

    gap_x = max(0.0, max(target_box[0] - reference_box[2], reference_box[0] - target_box[2]))
    gap_y = max(0.0, max(target_box[1] - reference_box[3], reference_box[1] - target_box[3]))
    edge_dist = math.sqrt(gap_x ** 2 + gap_y ** 2)
    edge_close = 1.0 - clamp01(edge_dist / max(diag * 0.35, 1.0))

    x_align = 1.0 - clamp01(dx / 0.35)
    y_align = 1.0 - clamp01(dy / 0.35)
    y_overlap = overlap_1d(target_box[1], target_box[3], reference_box[1], reference_box[3]) / max(min(th, rh), 1.0)
    horizontal_orientation = abs(tx - rx) / max(abs(tx - rx) + abs(ty - ry), 1e-6)

    if relation == "above":
        vertical = clamp01((ry - ty) / max(image_height * 0.35, 1.0))
        return 0.7 * vertical + 0.3 * x_align
    if relation == "below":
        vertical = clamp01((ty - ry) / max(image_height * 0.35, 1.0))
        return 0.7 * vertical + 0.3 * x_align
    if relation == "next to":
        return 0.4 * edge_close + 0.35 * y_align + 0.25 * horizontal_orientation
    if relation == "near":
        return 0.65 * edge_close + 0.2 * center_close + 0.15 * (0.5 * x_align + 0.5 * y_align)
    return 0.0


def run_prompt(processor, model, image: Image.Image, prompt: str, device: str, box_threshold: float, text_threshold: float) -> list[dict]:
    prompt = ensure_prompt_terminator(prompt)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
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
                "box_xyxy": [float(v) for v in box.tolist()],
                "score": float(score),
                "phrase": str(phrase),
            }
        )

    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions


def maybe_load_predictions(prediction_path: Path) -> list[dict] | None:
    if not prediction_path.exists():
        return None
    with open(prediction_path, "r", encoding="utf-8") as f:
        return json.load(f).get("predictions", [])


def filter_predictions_by_name(predictions: list[dict], object_name: str) -> list[dict]:
    expected = canonical_name(object_name)
    if not expected:
        return predictions
    filtered = [pred for pred in predictions if expected in canonical_name(pred.get("phrase", ""))]
    return filtered if filtered else predictions


def save_predictions(prediction_path: Path, sample_id: str, image_path: str, prompt: str, predictions: list[dict]) -> None:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prediction_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_id": sample_id,
                "image": image_path,
                "prompt": prompt,
                "predictions": predictions,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_benchmark(path: Path, keep_relations: set[str]) -> list[dict]:
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8", newline="")))
    filtered = [row for row in rows if row["normalized_relation"] in keep_relations]
    filtered.sort(key=lambda row: row["sample_id"])
    return filtered


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], key_prefix: str) -> dict:
    if not rows:
        return {"num_samples": 0}

    relation_groups = defaultdict(list)
    for row in rows:
        relation_groups[row["relation"]].append(row)

    overall_key = f"{key_prefix}_top1_acc_iou50"
    summary = {
        "num_samples": len(rows),
        overall_key: round(sum(int(row[f"{key_prefix}_top1_hit_iou50"]) for row in rows) / len(rows), 4),
    }

    if f"{key_prefix}_num_predictions" in rows[0]:
        summary[f"avg_{key_prefix}_num_predictions"] = round(
            sum(float(row[f"{key_prefix}_num_predictions"]) for row in rows) / len(rows), 4
        )

    per_relation = []
    for relation, group in sorted(relation_groups.items()):
        rel_row = {
            "relation": relation,
            "num_samples": len(group),
            overall_key: round(sum(int(row[f"{key_prefix}_top1_hit_iou50"]) for row in group) / len(group), 4),
        }
        if f"{key_prefix}_num_predictions" in rows[0]:
            rel_row[f"avg_{key_prefix}_num_predictions"] = round(
                sum(float(row[f"{key_prefix}_num_predictions"]) for row in group) / len(group), 4
            )
        per_relation.append(rel_row)
    summary["per_relation"] = per_relation
    return summary


def load_baseline_hits(path: Path, sample_ids: set[str]) -> dict[str, dict]:
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8", newline="")))
    return {row["sample_id"]: row for row in rows if row["sample_id"] in sample_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight geometry-based reranking experiment for spatial relations.")
    parser.add_argument("--benchmark-csv", required=True, help="CSV with local_image column.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-full-csv", default=None)
    parser.add_argument("--baseline-target-csv", default=None)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--relations", default="above,below,next to,near")
    parser.add_argument("--candidate-source", default="target_only", choices=["target_only", "full_prompt"])
    parser.add_argument("--reference-source", default="detected", choices=["detected", "gt"])
    parser.add_argument("--rerank-relations", default=None, help="Comma-separated subset of relations to actually rerank. Others keep the candidate top-1.")
    parser.add_argument("--w-target", type=float, default=0.45)
    parser.add_argument("--w-reference", type=float, default=0.15)
    parser.add_argument("--w-relation", type=float, default=0.40)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    keep_relations = {item.strip() for item in args.relations.split(",") if item.strip()}
    rerank_relations = keep_relations if args.rerank_relations is None else {item.strip() for item in args.rerank_relations.split(",") if item.strip()}
    benchmark_rows = load_benchmark(Path(args.benchmark_csv), keep_relations)
    if args.max_samples > 0:
        benchmark_rows = benchmark_rows[: args.max_samples]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable. Falling back to CPU.")
            device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Relations: {sorted(keep_relations)}")
    print(f"Rerank applied to: {sorted(rerank_relations)}")
    print(f"Candidate source: {args.candidate_source}")
    print(f"Reference source: {args.reference_source}")
    print(f"Weights: target={args.w_target}, reference={args.w_reference}, relation={args.w_relation}")

    processor = None
    model = None

    per_sample_rows = []

    for idx, row in enumerate(benchmark_rows, start=1):
        sample_id = row["sample_id"]
        image_path = row["local_image"]
        relation = row["normalized_relation"]
        candidate_prompt = row["prompt"] if args.candidate_source == "full_prompt" else f"the {row['target_name']}"
        reference_prompt = f"the {row['reference_name']}"

        candidate_prediction_path = output_dir / "candidates" / args.candidate_source / sample_id / "predictions.json"
        reference_prediction_path = output_dir / "candidates" / "reference_only" / sample_id / "predictions.json"

        print(f"[{idx}/{len(benchmark_rows)}] {sample_id}")

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

            candidate_predictions = maybe_load_predictions(candidate_prediction_path)
            if candidate_predictions is None:
                if processor is None or model is None:
                    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=True)
                    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id, local_files_only=True)
                    model.to(device)
                    model.eval()
                candidate_predictions = run_prompt(
                    processor,
                    model,
                    image,
                    candidate_prompt,
                    device,
                    args.box_threshold,
                    args.text_threshold,
                )
                save_predictions(candidate_prediction_path, sample_id, image_path, candidate_prompt, candidate_predictions)

            if args.reference_source == "detected":
                reference_predictions = maybe_load_predictions(reference_prediction_path)
                if reference_predictions is None:
                    if processor is None or model is None:
                        processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=True)
                        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id, local_files_only=True)
                        model.to(device)
                        model.eval()
                    reference_predictions = run_prompt(
                        processor,
                        model,
                        image,
                        reference_prompt,
                        device,
                        args.box_threshold,
                        args.text_threshold,
                    )
                    save_predictions(reference_prediction_path, sample_id, image_path, reference_prompt, reference_predictions)
            else:
                reference_predictions = [
                    {
                        "box_xyxy": json.loads(row["reference_bbox_xyxy"]),
                        "score": 1.0,
                        "phrase": row["reference_name"],
                    }
                ]

        candidate_predictions = filter_predictions_by_name(candidate_predictions, row["target_name"])
        reference_predictions = filter_predictions_by_name(reference_predictions, row["reference_name"])

        gt_box = json.loads(row["target_bbox_xyxy"])
        image_width = float(row["image_width"])
        image_height = float(row["image_height"])

        candidate_top = candidate_predictions[0] if candidate_predictions else None
        candidate_top_iou = iou_xyxy(candidate_top["box_xyxy"], gt_box) if candidate_top else 0.0

        max_target_score = max((pred["score"] for pred in candidate_predictions), default=1.0)
        max_reference_score = max((pred["score"] for pred in reference_predictions), default=1.0)

        target_name_same_as_reference = canonical_name(row["target_name"]) == canonical_name(row["reference_name"])
        best_pair = None

        if relation not in rerank_relations:
            best_pair = {
                "target": candidate_top,
                "reference": None,
                "reference_conf": 0.0,
                "relation_score": 0.0,
                "final_score": candidate_top["score"] if candidate_top else 0.0,
            }
        else:
            for target_pred in candidate_predictions:
                target_conf = target_pred["score"] / max(max_target_score, 1e-6)
                if not reference_predictions:
                    pair = {
                        "target": target_pred,
                        "reference": None,
                        "reference_conf": 0.0,
                        "relation_score": 0.0,
                        "final_score": args.w_target * target_conf,
                    }
                    if best_pair is None or pair["final_score"] > best_pair["final_score"]:
                        best_pair = pair
                    continue

                for reference_pred in reference_predictions:
                    if target_name_same_as_reference and iou_xyxy(target_pred["box_xyxy"], reference_pred["box_xyxy"]) >= 0.9:
                        continue
                    reference_conf = reference_pred["score"] / max(max_reference_score, 1e-6)
                    rel_score = relation_score(
                        target_pred["box_xyxy"],
                        reference_pred["box_xyxy"],
                        relation,
                        image_width,
                        image_height,
                    )
                    final_score = (
                        args.w_target * target_conf
                        + args.w_reference * reference_conf
                        + args.w_relation * rel_score
                    )
                    pair = {
                        "target": target_pred,
                        "reference": reference_pred,
                        "reference_conf": reference_conf,
                        "relation_score": rel_score,
                        "final_score": final_score,
                    }
                    if best_pair is None or pair["final_score"] > best_pair["final_score"]:
                        best_pair = pair

        reranked_target = best_pair["target"] if best_pair else None
        reranked_iou = iou_xyxy(reranked_target["box_xyxy"], gt_box) if reranked_target else 0.0

        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "image_id": row["image_id"],
                "relation": relation,
                "target_name": row["target_name"],
                "reference_name": row["reference_name"],
                "candidate_source": args.candidate_source,
                "candidate_num_predictions": len(candidate_predictions),
                "reference_only_num_predictions": len(reference_predictions),
                "candidate_top1_iou": round(candidate_top_iou, 4),
                "candidate_top1_hit_iou50": int(candidate_top_iou >= args.iou_threshold),
                "candidate_top1_phrase": candidate_top["phrase"] if candidate_top else "",
                "candidate_top1_score": round(candidate_top["score"], 4) if candidate_top else 0.0,
                "reranked_top1_iou": round(reranked_iou, 4),
                "reranked_top1_hit_iou50": int(reranked_iou >= args.iou_threshold),
                "reranked_target_phrase": reranked_target["phrase"] if reranked_target else "",
                "reranked_target_score": round(reranked_target["score"], 4) if reranked_target else 0.0,
                "reranked_reference_phrase": best_pair["reference"]["phrase"] if best_pair and best_pair["reference"] else "",
                "reranked_reference_score": round(best_pair["reference"]["score"], 4) if best_pair and best_pair["reference"] else 0.0,
                "reranked_relation_score": round(best_pair["relation_score"], 4) if best_pair else 0.0,
                "reranked_final_score": round(best_pair["final_score"], 4) if best_pair else 0.0,
            }
        )

    per_sample_csv = output_dir / "reranking_per_sample.csv"
    write_csv(per_sample_csv, per_sample_rows)

    summary = {
        "model_id": args.model_id,
        "device": device,
        "relations": sorted(keep_relations),
        "candidate_source": args.candidate_source,
        "reference_source": args.reference_source,
        "rerank_relations": sorted(rerank_relations),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "weights": {
            "target": args.w_target,
            "reference": args.w_reference,
            "relation": args.w_relation,
        },
        "num_samples": len(per_sample_rows),
        "candidate": summarize(per_sample_rows, "candidate"),
        "reranked": summarize(per_sample_rows, "reranked"),
        "avg_reference_only_num_predictions": round(
            sum(row["reference_only_num_predictions"] for row in per_sample_rows) / len(per_sample_rows), 4
        ) if per_sample_rows else 0.0,
        "num_samples_with_multi_candidate_predictions": sum(row["candidate_num_predictions"] >= 2 for row in per_sample_rows),
        "num_samples_with_no_reference_candidates": sum(row["reference_only_num_predictions"] == 0 for row in per_sample_rows),
    }

    sample_ids = {row["sample_id"] for row in per_sample_rows}
    if args.baseline_full_csv:
        full_baseline = load_baseline_hits(Path(args.baseline_full_csv), sample_ids)
        summary["previous_full_prompt_baseline_top1_acc_iou50"] = round(
            sum(int(full_baseline[sample_id]["top1_hit_iou50"]) for sample_id in sample_ids) / len(sample_ids),
            4,
        )
    if args.baseline_target_csv:
        prev_target = load_baseline_hits(Path(args.baseline_target_csv), sample_ids)
        summary["previous_target_only_baseline_top1_acc_iou50"] = round(
            sum(int(prev_target[sample_id]["top1_hit_iou50"]) for sample_id in sample_ids) / len(sample_ids),
            4,
        )

    multi_candidate_rows = [row for row in per_sample_rows if row["candidate_num_predictions"] >= 2]
    if multi_candidate_rows:
        summary["multi_candidate_subset"] = {
            "num_samples": len(multi_candidate_rows),
            "candidate_top1_acc_iou50": round(
                sum(row["candidate_top1_hit_iou50"] for row in multi_candidate_rows) / len(multi_candidate_rows),
                4,
            ),
            "reranked_top1_acc_iou50": round(
                sum(row["reranked_top1_hit_iou50"] for row in multi_candidate_rows) / len(multi_candidate_rows),
                4,
            ),
            "avg_reference_only_num_predictions": round(
                sum(row["reference_only_num_predictions"] for row in multi_candidate_rows) / len(multi_candidate_rows),
                4,
            ),
        }

    with open(output_dir / "reranking_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
