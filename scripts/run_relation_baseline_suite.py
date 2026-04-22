import argparse
import csv
import json
import os
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


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


def prompt_for_variant(row: dict, variant: str) -> str:
    if variant == "full_prompt":
        return row["prompt"]
    if variant == "target_only":
        return f"the {row['target_name']}"
    if variant == "counterfactual":
        relation = COUNTERFACTUAL_RELATION_MAP[row["normalized_relation"]]
        return f"the {row['target_name']} {relation} the {row['reference_name']}"
    raise ValueError(f"Unsupported variant: {variant}")


def bbox_xyxy_from_xywh(obj: dict) -> list[float]:
    x = float(obj.get("x", 0))
    y = float(obj.get("y", 0))
    w = float(obj.get("w", 0))
    h = float(obj.get("h", 0))
    return [x, y, x + w, y + h]


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


def download_image(url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return False
    urllib.request.urlretrieve(url, out_path)
    return True


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def attach_local_images(rows: list[dict], images_dir: Path) -> tuple[list[dict], dict]:
    counts = {"downloaded_now": 0, "reused_existing": 0}
    seen_urls = {}
    for row in rows:
        url = row["image_url"]
        if url in seen_urls:
            row["local_image"] = seen_urls[url]
            continue

        filename = os.path.basename(url)
        local_path = images_dir / filename
        changed = download_image(url, local_path)
        row["local_image"] = str(local_path)
        seen_urls[url] = str(local_path)
        if changed:
            counts["downloaded_now"] += 1
        else:
            counts["reused_existing"] += 1

    counts["unique_images"] = len(seen_urls)
    return rows, counts


def move_inputs_to_device(inputs, device: str):
    if device == "cpu":
        return inputs
    return inputs.to(device)


def run_prompt(processor, model, image: Image.Image, prompt: str, device: str, box_threshold: float, text_threshold: float) -> list[dict]:
    prompt = ensure_prompt_terminator(prompt)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, device)

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


def evaluate_predictions(row: dict, variant: str, prompt: str, predictions: list[dict], iou_threshold: float) -> dict:
    gt_box = json.loads(row["target_bbox_xyxy"])
    target_name = row["target_name"]
    target_name_canonical = canonical_name(target_name)

    scored = []
    for pred in predictions:
        pred_box = pred["box_xyxy"]
        phrase = pred["phrase"]
        scored.append(
            {
                "box_xyxy": pred_box,
                "score": float(pred["score"]),
                "phrase": phrase,
                "iou_with_target": iou_xyxy(pred_box, gt_box),
                "matches_target_phrase": target_name_canonical in canonical_name(phrase),
            }
        )

    num_predictions = len(scored)
    top1 = scored[0] if scored else None
    top1_iou = top1["iou_with_target"] if top1 else 0.0
    best_iou_any = max((item["iou_with_target"] for item in scored), default=0.0)
    best_iou_target_phrase = max((item["iou_with_target"] for item in scored if item["matches_target_phrase"]), default=0.0)
    num_hit_predictions = sum(1 for item in scored if item["iou_with_target"] >= iou_threshold)
    error_type = classify_error(num_predictions, top1_iou, best_iou_any, iou_threshold)

    return {
        "sample_id": row["sample_id"],
        "image_id": row["image_id"],
        "relation": row["normalized_relation"],
        "prompt_variant": variant,
        "prompt_used": prompt,
        "target_name": row["target_name"],
        "reference_name": row["reference_name"],
        "num_predictions": num_predictions,
        "num_hit_predictions_iou50": num_hit_predictions,
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


def save_predictions(output_root: Path, variant: str, sample_id: str, image_path: str, prompt: str, predictions: list[dict]) -> None:
    sample_dir = output_root / variant / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_dir / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
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
        return {
            "num_samples": 0,
            "iou_threshold": iou_threshold,
        }

    relation_groups = defaultdict(list)
    error_counter = Counter()
    for row in rows:
        relation_groups[row["relation"]].append(row)
        error_counter[row["error_type"]] += 1

    relation_summary = []
    for relation, group in sorted(relation_groups.items()):
        total = len(group)
        relation_summary.append(
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
        "per_relation": relation_summary,
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


def open_json_stream(path: Path):
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path, "r")
        members = [name for name in archive.namelist() if name.lower().endswith(".json")]
        if not members:
            archive.close()
            raise FileNotFoundError(f"No JSON file found inside {path}")
        return archive.open(members[0], "r"), archive
    return open(path, "rb"), None


def load_relevant_objects(objects_path: Path, relevant_image_ids: set[int]) -> dict[int, list[dict]]:
    object_index = {}
    stream, archive = open_json_stream(objects_path)
    try:
        for item in ijson.items(stream, "item"):
            image_id = int(item.get("image_id", -1))
            if image_id in relevant_image_ids:
                object_index[image_id] = item.get("objects", [])
                if len(object_index) == len(relevant_image_ids):
                    break
    finally:
        stream.close()
        if archive is not None:
            archive.close()
    return object_index


def load_relevant_objects_from_api(relevant_image_ids: set[int]) -> dict[int, list[dict]]:
    object_index = {}
    sorted_ids = sorted(relevant_image_ids)
    total = len(sorted_ids)
    for idx, image_id in enumerate(sorted_ids, start=1):
        print(f"[chance-api {idx}/{total}] image_id={image_id}")
        try:
            payload = fetch_json(f"https://visualgenome.org/api/v0/images/{image_id}/graph")
        except Exception as exc:
            print(f"Warning: failed to fetch graph for image {image_id}: {exc}")
            object_index[image_id] = []
            continue

        converted_objects = []
        for bbox in payload.get("bounding_boxes", []):
            converted_objects.append(
                {
                    "x": bbox.get("x", 0),
                    "y": bbox.get("y", 0),
                    "w": bbox.get("width", 0),
                    "h": bbox.get("height", 0),
                    "names": [boxed.get("name", "") for boxed in bbox.get("boxed_objects", []) if boxed.get("name")],
                }
            )
        object_index[image_id] = converted_objects
    return object_index


def object_name_variants(obj: dict) -> set[str]:
    variants = set()
    for name in obj.get("names", []):
        canonical = canonical_name(name)
        if canonical:
            variants.add(canonical)
    if not variants and obj.get("name"):
        canonical = canonical_name(obj["name"])
        if canonical:
            variants.add(canonical)
    return variants


def compute_chance_baseline(rows: list[dict], iou_threshold: float, objects_path: Path | None = None, use_vg_api: bool = False) -> tuple[list[dict], dict]:
    relevant_image_ids = {int(row["image_id"]) for row in rows}
    if use_vg_api:
        object_index = load_relevant_objects_from_api(relevant_image_ids)
    elif objects_path is not None:
        object_index = load_relevant_objects(objects_path, relevant_image_ids)
    else:
        raise ValueError("Either objects_path must be provided or use_vg_api must be True.")

    chance_rows = []
    for row in rows:
        image_id = int(row["image_id"])
        target_name = canonical_name(row["target_name"])
        gt_box = json.loads(row["target_bbox_xyxy"])
        candidates = []

        for obj in object_index.get(image_id, []):
            if target_name in object_name_variants(obj):
                candidates.append(obj)

        hits = 0
        for candidate in candidates:
            if iou_xyxy(bbox_xyxy_from_xywh(candidate), gt_box) >= iou_threshold:
                hits += 1

        num_candidates = len(candidates)
        chance = (hits / num_candidates) if num_candidates else 0.0
        chance_rows.append(
            {
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "relation": row["normalized_relation"],
                "target_name": row["target_name"],
                "num_target_gt_candidates": num_candidates,
                "num_target_gt_hits_iou50": hits,
                "chance_top1_acc_iou50": round(chance, 4),
                "relation_necessary": int(num_candidates >= 2),
            }
        )

    total = len(chance_rows)
    relation_groups = defaultdict(list)
    for row in chance_rows:
        relation_groups[row["relation"]].append(row)

    per_relation = []
    for relation, group in sorted(relation_groups.items()):
        per_relation.append(
            {
                "relation": relation,
                "num_samples": len(group),
                "chance_top1_acc_iou50": round(sum(item["chance_top1_acc_iou50"] for item in group) / len(group), 4),
                "avg_num_target_gt_candidates": round(sum(item["num_target_gt_candidates"] for item in group) / len(group), 4),
            }
        )

    summary = {
        "num_samples": total,
        "iou_threshold": iou_threshold,
        "chance_top1_acc_iou50": round(sum(row["chance_top1_acc_iou50"] for row in chance_rows) / total, 4) if total else 0.0,
        "avg_num_target_gt_candidates": round(sum(row["num_target_gt_candidates"] for row in chance_rows) / total, 4) if total else 0.0,
        "num_relation_necessary_samples": sum(row["relation_necessary"] for row in chance_rows),
        "per_relation": per_relation,
    }
    return chance_rows, summary


def compute_random_from_target_only_baseline(target_only_rows: list[dict]) -> tuple[list[dict], dict]:
    baseline_rows = []
    for row in target_only_rows:
        num_candidates = int(row["num_predictions"])
        num_hits = int(row["num_hit_predictions_iou50"])
        chance = (num_hits / num_candidates) if num_candidates else 0.0
        baseline_rows.append(
            {
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "relation": row["relation"],
                "target_name": row["target_name"],
                "num_target_only_candidates": num_candidates,
                "num_target_only_hits_iou50": num_hits,
                "approx_chance_top1_acc_iou50": round(chance, 4),
                "relation_necessary_proxy": int(num_candidates >= 2),
            }
        )

    total = len(baseline_rows)
    relation_groups = defaultdict(list)
    for row in baseline_rows:
        relation_groups[row["relation"]].append(row)

    per_relation = []
    for relation, group in sorted(relation_groups.items()):
        per_relation.append(
            {
                "relation": relation,
                "num_samples": len(group),
                "approx_chance_top1_acc_iou50": round(sum(item["approx_chance_top1_acc_iou50"] for item in group) / len(group), 4),
                "avg_num_target_only_candidates": round(sum(item["num_target_only_candidates"] for item in group) / len(group), 4),
            }
        )

    summary = {
        "num_samples": total,
        "approx_chance_top1_acc_iou50": round(sum(row["approx_chance_top1_acc_iou50"] for row in baseline_rows) / total, 4) if total else 0.0,
        "avg_num_target_only_candidates": round(sum(row["num_target_only_candidates"] for row in baseline_rows) / total, 4) if total else 0.0,
        "num_relation_necessary_proxy_samples": sum(row["relation_necessary_proxy"] for row in baseline_rows),
        "per_relation": per_relation,
    }
    return baseline_rows, summary


def filter_rows_by_ids(rows: list[dict], sample_ids: set[str]) -> list[dict]:
    return [row for row in rows if row["sample_id"] in sample_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run relation-sensitivity baseline experiments on the Visual Genome relation benchmark.")
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--objects-json", default=None, help="Optional Visual Genome objects.json or objects.json.zip for the true chance baseline.")
    parser.add_argument("--use-vg-api-for-chance", action="store_true", help="Fetch scene graphs from the Visual Genome API instead of reading a local objects.json file.")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    benchmark_csv = Path(args.benchmark_csv)
    output_dir = Path(args.output_dir)
    images_dir = Path(args.images_dir) if args.images_dir else output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_benchmark(benchmark_csv)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    rows, image_stats = attach_local_images(rows, images_dir)
    write_csv(output_dir / "benchmark_with_local_images.csv", rows)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable. Falling back to CPU.")
            device = "cpu"

    print(f"Loading model: {args.model_id}")
    print(f"Using device: {device}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    variant_rows = {variant: [] for variant in VARIANTS}
    total_samples = len(rows)

    for idx, row in enumerate(rows, start=1):
        image_path = Path(row["local_image"])
        print(f"[{idx}/{total_samples}] {row['sample_id']}")
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            for variant in VARIANTS:
                prompt = prompt_for_variant(row, variant)
                predictions = run_prompt(
                    processor=processor,
                    model=model,
                    image=image,
                    prompt=prompt,
                    device=device,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )
                save_predictions(output_dir, variant, row["sample_id"], str(image_path), prompt, predictions)
                variant_rows[variant].append(
                    evaluate_predictions(
                        row=row,
                        variant=variant,
                        prompt=prompt,
                        predictions=predictions,
                        iou_threshold=args.iou_threshold,
                    )
                )

    variant_summaries = {}
    for variant, result_rows in variant_rows.items():
        write_csv(output_dir / f"{variant}_per_sample.csv", result_rows)
        summary = summarize_variant_results(result_rows, args.iou_threshold)
        variant_summaries[variant] = summary
        with open(output_dir / f"{variant}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    chance_rows = []
    chance_summary = None
    relation_necessary_ids = set()
    if args.objects_json or args.use_vg_api_for_chance:
        chance_rows, chance_summary = compute_chance_baseline(
            rows,
            args.iou_threshold,
            objects_path=Path(args.objects_json) if args.objects_json else None,
            use_vg_api=args.use_vg_api_for_chance,
        )
        write_csv(output_dir / "chance_baseline_per_sample.csv", chance_rows)
        with open(output_dir / "chance_baseline_summary.json", "w", encoding="utf-8") as f:
            json.dump(chance_summary, f, indent=2, ensure_ascii=False)
        relation_necessary_ids = {row["sample_id"] for row in chance_rows if int(row["relation_necessary"]) == 1}

    approx_chance_rows, approx_chance_summary = compute_random_from_target_only_baseline(variant_rows["target_only"])
    write_csv(output_dir / "approx_random_from_target_only_per_sample.csv", approx_chance_rows)
    with open(output_dir / "approx_random_from_target_only_summary.json", "w", encoding="utf-8") as f:
        json.dump(approx_chance_summary, f, indent=2, ensure_ascii=False)
    if not relation_necessary_ids:
        relation_necessary_ids = {row["sample_id"] for row in approx_chance_rows if int(row["relation_necessary_proxy"]) == 1}

    subset_summaries = {}
    if relation_necessary_ids:
        for variant, result_rows in variant_rows.items():
            subset_summaries[variant] = summarize_variant_results(
                filter_rows_by_ids(result_rows, relation_necessary_ids),
                args.iou_threshold,
            )
        subset_summaries["approx_random_from_target_only"] = {
            "num_samples": len(relation_necessary_ids),
            "approx_chance_top1_acc_iou50": round(
                sum(row["approx_chance_top1_acc_iou50"] for row in approx_chance_rows if row["sample_id"] in relation_necessary_ids) / len(relation_necessary_ids),
                4,
            ),
            "avg_num_target_only_candidates": round(
                sum(row["num_target_only_candidates"] for row in approx_chance_rows if row["sample_id"] in relation_necessary_ids) / len(relation_necessary_ids),
                4,
            ),
        }
        if chance_rows:
            subset_summaries["chance_baseline"] = {
                "num_samples": len(relation_necessary_ids),
                "chance_top1_acc_iou50": round(
                    sum(row["chance_top1_acc_iou50"] for row in chance_rows if row["sample_id"] in relation_necessary_ids) / len(relation_necessary_ids),
                    4,
                ),
                "avg_num_target_gt_candidates": round(
                    sum(row["num_target_gt_candidates"] for row in chance_rows if row["sample_id"] in relation_necessary_ids) / len(relation_necessary_ids),
                    4,
                ),
            }

    suite_summary = {
        "model_id": args.model_id,
        "device": device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "iou_threshold": args.iou_threshold,
        "num_samples": total_samples,
        "image_download": image_stats,
        "counterfactual_relation_map": COUNTERFACTUAL_RELATION_MAP,
        "variants": variant_summaries,
        "comparisons": {
            "full_vs_target_only": compare_result_sets(variant_rows["full_prompt"], variant_rows["target_only"]),
            "full_vs_counterfactual": compare_result_sets(variant_rows["full_prompt"], variant_rows["counterfactual"]),
        },
        "approx_random_from_target_only": approx_chance_summary,
    }

    if chance_summary is not None:
        suite_summary["chance_baseline"] = chance_summary
    if subset_summaries:
        suite_summary["relation_necessary_subset"] = subset_summaries

    with open(output_dir / "suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(suite_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
