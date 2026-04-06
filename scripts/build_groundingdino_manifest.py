import argparse
import csv
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    rows = []
    kept = 0
    skipped = 0

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row.get("local_image", "").strip()
            sample_id = row.get("sample_id", "").strip()
            prompt = row.get("prompt", "").strip()

            if not image_path or not sample_id or not prompt or not os.path.exists(image_path):
                skipped += 1
                continue

            rows.append({
                "sample_id": sample_id,
                "image": image_path,
                "prompt": prompt,
            })
            kept += 1

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved manifest to: {args.output_csv}")
    print(f"Kept rows: {kept}")
    print(f"Skipped rows: {skipped}")


if __name__ == "__main__":
    main()
