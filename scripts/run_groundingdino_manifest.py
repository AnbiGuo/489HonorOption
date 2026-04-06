import argparse
import csv
import json
import os
from pathlib import Path

import cv2
from groundingdino.util.inference import annotate, load_image, load_model, predict


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO on every row in a CSV manifest."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="CSV with columns: sample_id,image,prompt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.config, args.checkpoint, device=args.device)

    summary_rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            sample_id = row.get("sample_id") or f"sample_{idx:03d}"
            image_path = row["image"]
            prompt = row["prompt"]

            sample_dir = output_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            image_source, image = load_image(image_path)
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.device,
            )

            annotated_frame = annotate(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=phrases,
            )

            annotated_path = sample_dir / "annotated.jpg"
            prediction_path = sample_dir / "predictions.json"
            cv2.imwrite(str(annotated_path), annotated_frame)

            predictions = []
            for box, logit, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
                predictions.append(
                    {
                        "box_xyxy": box,
                        "score": float(logit),
                        "phrase": phrase,
                    }
                )

            with open(prediction_path, "w", encoding="utf-8") as out_f:
                json.dump(
                    {
                        "sample_id": sample_id,
                        "image": image_path,
                        "prompt": prompt,
                        "box_threshold": args.box_threshold,
                        "text_threshold": args.text_threshold,
                        "device": args.device,
                        "predictions": predictions,
                    },
                    out_f,
                    indent=2,
                    ensure_ascii=False,
                )

            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "image": image_path,
                    "prompt": prompt,
                    "num_predictions": len(predictions),
                    "annotated": str(annotated_path),
                    "predictions_json": str(prediction_path),
                }
            )
            print(f"[{idx}] {sample_id}: {len(predictions)} predictions")

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "image",
                "prompt",
                "num_predictions",
                "annotated",
                "predictions_json",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
