import argparse
import csv
import json
import random
from pathlib import Path

RELATION_MAP = {
    "above": "above",
    "over": "above",
    "on top of": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
    "behind": "behind",
    "in front of": "in front of",
    "on front of": "in front of",
    "next to": "next to",
    "beside": "next to",
    "near": "near",
}

ALLOWED_RELATIONS = [
    "above",
    "below",
    "behind",
    "in front of",
    "next to",
    "near",
]


def clean_name(obj):
    names = obj.get("names") or []
    if names:
        return str(names[0]).strip().lower()
    name = obj.get("name")
    return str(name).strip().lower() if name else ""


def bbox_xyxy(obj):
    x = float(obj.get("x", 0))
    y = float(obj.get("y", 0))
    w = float(obj.get("w", 0))
    h = float(obj.get("h", 0))
    return [x, y, x + w, y + h]


def area(obj):
    return float(obj.get("w", 0)) * float(obj.get("h", 0))


def main():
    parser = argparse.ArgumentParser(description="Build a normalized spatial-relation benchmark from Visual Genome relationships.")
    parser.add_argument("--relationships-json", required=True)
    parser.add_argument("--image-data-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--min-box-area", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.relationships_json, "r", encoding="utf-8") as f:
        relationships = json.load(f)

    with open(args.image_data_json, "r", encoding="utf-8") as f:
        image_data = json.load(f)

    image_meta = {int(item["image_id"]): item for item in image_data}

    rows = []
    per_relation_counts = {rel: 0 for rel in ALLOWED_RELATIONS}

    for item in relationships:
        image_id = int(item.get("image_id"))
        meta = image_meta.get(image_id)
        if not meta:
            continue

        image_url = meta.get("url", "")
        width = meta.get("width")
        height = meta.get("height")

        for rel in item.get("relationships", []):
            raw_predicate = str(rel.get("predicate", "")).strip().lower()
            normalized = RELATION_MAP.get(raw_predicate)
            if normalized is None:
                continue

            subject = rel.get("subject", {})
            obj = rel.get("object", {})

            target_name = clean_name(subject)
            reference_name = clean_name(obj)
            if not target_name or not reference_name:
                continue

            if area(subject) < args.min_box_area or area(obj) < args.min_box_area:
                continue

            target_bbox = bbox_xyxy(subject)
            reference_bbox = bbox_xyxy(obj)

            prompt = f"the {target_name} {normalized} the {reference_name}"
            sample_id = f"vg_{image_id}_{rel.get('relationship_id')}"

            rows.append(
                {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "image_url": image_url,
                    "image_width": width,
                    "image_height": height,
                    "target_name": target_name,
                    "reference_name": reference_name,
                    "raw_predicate": raw_predicate,
                    "normalized_relation": normalized,
                    "prompt": prompt,
                    "target_bbox_xyxy": json.dumps(target_bbox),
                    "reference_bbox_xyxy": json.dumps(reference_bbox),
                }
            )
            per_relation_counts[normalized] += 1

    rows.sort(key=lambda r: (r["normalized_relation"], r["image_id"], r["sample_id"]))

    if args.max_samples and len(rows) > args.max_samples:
        buckets = {rel: [] for rel in ALLOWED_RELATIONS}
        for row in rows:
            buckets[row["normalized_relation"]].append(row)

        selected = []
        per_bucket = max(1, args.max_samples // len(ALLOWED_RELATIONS))
        for rel in ALLOWED_RELATIONS:
            bucket = buckets[rel]
            random.shuffle(bucket)
            selected.extend(bucket[:per_bucket])

        if len(selected) < args.max_samples:
            leftovers = []
            selected_ids = {r["sample_id"] for r in selected}
            for row in rows:
                if row["sample_id"] not in selected_ids:
                    leftovers.append(row)
            random.shuffle(leftovers)
            selected.extend(leftovers[: args.max_samples - len(selected)])

        rows = selected[: args.max_samples]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "image_id",
                "image_url",
                "image_width",
                "image_height",
                "target_name",
                "reference_name",
                "raw_predicate",
                "normalized_relation",
                "prompt",
                "target_bbox_xyxy",
                "reference_bbox_xyxy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    final_counts = {rel: 0 for rel in ALLOWED_RELATIONS}
    for row in rows:
        final_counts[row["normalized_relation"]] += 1

    print(f"Saved benchmark to: {output_path}")
    print(f"Total rows: {len(rows)}")
    print("Counts by normalized relation:")
    for rel in ALLOWED_RELATIONS:
        print(f"  {rel}: {final_counts[rel]}")


if __name__ == "__main__":
    main()
