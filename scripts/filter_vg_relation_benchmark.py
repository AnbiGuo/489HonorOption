import argparse
import csv
from pathlib import Path

CLEAN_ALLOWED = {
    "above": {"above"},
    "below": {"below"},
    "behind": {"behind"},
    "in front of": {"in front of"},
    "next to": {"next to", "beside"},
    "near": {"near"},
}

ORDER = ["above", "below", "behind", "in front of", "next to", "near"]


def main():
    parser = argparse.ArgumentParser(description="Filter a Visual Genome relation benchmark into a cleaner v1 subset.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-samples", type=int, default=120)
    args = parser.parse_args()

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    buckets = {rel: [] for rel in ORDER}
    for row in rows:
        normalized = row["normalized_relation"]
        raw = row["raw_predicate"].strip().lower()
        allowed = CLEAN_ALLOWED.get(normalized)
        if allowed is None:
            continue
        if raw not in allowed:
            continue
        buckets[normalized].append(row)

    per_bucket = max(1, args.max_samples // len(ORDER))
    selected = []
    for rel in ORDER:
        bucket = buckets[rel]
        bucket.sort(key=lambda r: (int(r["image_id"]), r["sample_id"]))
        selected.extend(bucket[:per_bucket])

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Saved clean benchmark to: {output_path}")
    print(f"Total rows: {len(selected)}")
    for rel in ORDER:
        print(f"  {rel}: {sum(1 for r in selected if r['normalized_relation'] == rel)}")


if __name__ == "__main__":
    main()
