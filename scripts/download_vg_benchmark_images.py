import argparse
import csv
import os
import urllib.request
from pathlib import Path


def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return False
    urllib.request.urlretrieve(url, out_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download only the Visual Genome images referenced in a benchmark CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-csv", required=True, help="CSV copy with local_image path added")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    if "local_image" not in fieldnames:
        fieldnames.append("local_image")

    seen = {}
    downloaded = 0
    reused = 0

    for row in rows:
        url = row["image_url"]
        filename = os.path.basename(url)
        local_path = images_dir / filename

        if url not in seen:
            changed = download(url, local_path)
            seen[url] = str(local_path)
            if changed:
                downloaded += 1
            else:
                reused += 1

        row["local_image"] = seen[url]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved local benchmark CSV to: {output_csv}")
    print(f"Unique images: {len(seen)}")
    print(f"Downloaded now: {downloaded}")
    print(f"Reused existing: {reused}")


if __name__ == "__main__":
    main()
