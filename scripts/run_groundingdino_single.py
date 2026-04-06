import argparse
import json
import os

import cv2
from groundingdino.util.inference import annotate, load_image, load_model, predict


def main():
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO on a single image and save both visualization and JSON output."
    )
    parser.add_argument("--config", required=True, help="Path to model config .py file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pth file")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--text", required=True, help="Text prompt, e.g. 'person . chair .'")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.config, args.checkpoint, device=args.device)
    image_source, image = load_image(args.image)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=args.text,
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

    image_out = os.path.join(args.output_dir, "annotated.jpg")
    json_out = os.path.join(args.output_dir, "predictions.json")

    cv2.imwrite(image_out, annotated_frame)

    results = []
    for box, logit, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
        results.append(
            {
                "box_xyxy": box,
                "score": float(logit),
                "phrase": phrase,
            }
        )

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": args.image,
                "text": args.text,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "device": args.device,
                "predictions": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved image to: {image_out}")
    print(f"Saved predictions to: {json_out}")
    print(f"Num predictions: {len(results)}")


if __name__ == "__main__":
    main()
