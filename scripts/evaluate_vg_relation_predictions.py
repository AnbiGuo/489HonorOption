import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def cxcywh_norm_to_xyxy(box, width, height):
    cx, cy, w, h = box
    cx *= width
    cy *= height
    w *= width
    h *= height
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def iou_xyxy(a, b):
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


def load_benchmark(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return {row['sample_id']: row for row in rows}


def classify_error(num_predictions, top1_iou, best_iou_any, iou_threshold):
    if num_predictions == 0:
        return 'no_prediction'
    if top1_iou >= iou_threshold:
        return 'top1_correct'
    if best_iou_any >= iou_threshold:
        return 'found_but_not_top1'
    return 'wrong_localization'


def main():
    parser = argparse.ArgumentParser(description='Evaluate GroundingDINO predictions against the Visual Genome relation benchmark.')
    parser.add_argument('--benchmark-csv', required=True)
    parser.add_argument('--predictions-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark_csv)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows = []
    relation_stats = defaultdict(list)
    error_counter = Counter()

    for sample_id, row in benchmark.items():
        pred_file = predictions_dir / sample_id / 'predictions.json'
        pred_payload = None
        preds = []
        if pred_file.exists():
            with open(pred_file, 'r', encoding='utf-8') as f:
                pred_payload = json.load(f)
            preds = pred_payload.get('predictions', [])

        width = float(row['image_width'])
        height = float(row['image_height'])
        gt_box = json.loads(row['target_bbox_xyxy'])
        relation = row['normalized_relation']
        target_name = row['target_name']

        converted = []
        for pred in preds:
            xyxy = cxcywh_norm_to_xyxy(pred['box_xyxy'], width, height)
            converted.append(
                {
                    'xyxy': xyxy,
                    'score': float(pred['score']),
                    'phrase': pred['phrase'],
                    'iou_with_target': iou_xyxy(xyxy, gt_box),
                }
            )

        converted.sort(key=lambda x: x['score'], reverse=True)
        num_predictions = len(converted)

        top1_iou = converted[0]['iou_with_target'] if converted else 0.0
        top1_score = converted[0]['score'] if converted else 0.0
        top1_phrase = converted[0]['phrase'] if converted else ''
        best_iou_any = max((p['iou_with_target'] for p in converted), default=0.0)
        phrase_filtered = [p for p in converted if target_name in p['phrase'].lower()]
        best_iou_target_phrase = max((p['iou_with_target'] for p in phrase_filtered), default=0.0)

        top1_hit = top1_iou >= args.iou_threshold
        any_hit = best_iou_any >= args.iou_threshold
        target_phrase_hit = best_iou_target_phrase >= args.iou_threshold
        error_type = classify_error(num_predictions, top1_iou, best_iou_any, args.iou_threshold)
        error_counter[error_type] += 1

        sample_result = {
            'sample_id': sample_id,
            'image_id': row['image_id'],
            'relation': relation,
            'prompt': row['prompt'],
            'target_name': target_name,
            'reference_name': row['reference_name'],
            'num_predictions': num_predictions,
            'top1_iou': round(top1_iou, 4),
            'best_iou_any': round(best_iou_any, 4),
            'best_iou_target_phrase': round(best_iou_target_phrase, 4),
            'top1_score': round(top1_score, 4),
            'top1_phrase': top1_phrase,
            'top1_hit_iou50': int(top1_hit),
            'any_hit_iou50': int(any_hit),
            'target_phrase_hit_iou50': int(target_phrase_hit),
            'error_type': error_type,
            'predictions_json': str(pred_file),
        }
        per_sample_rows.append(sample_result)
        relation_stats[relation].append(sample_result)

    per_sample_csv = output_dir / 'per_sample_results.csv'
    with open(per_sample_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)

    relation_summary_rows = []
    for relation, rows in sorted(relation_stats.items()):
        total = len(rows)
        relation_summary_rows.append(
            {
                'relation': relation,
                'num_samples': total,
                'top1_acc_iou50': round(sum(r['top1_hit_iou50'] for r in rows) / total, 4),
                'any_hit_acc_iou50': round(sum(r['any_hit_iou50'] for r in rows) / total, 4),
                'target_phrase_acc_iou50': round(sum(r['target_phrase_hit_iou50'] for r in rows) / total, 4),
                'avg_num_predictions': round(sum(r['num_predictions'] for r in rows) / total, 4),
            }
        )

    relation_summary_csv = output_dir / 'relation_summary.csv'
    with open(relation_summary_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['relation', 'num_samples', 'top1_acc_iou50', 'any_hit_acc_iou50', 'target_phrase_acc_iou50', 'avg_num_predictions'],
        )
        writer.writeheader()
        writer.writerows(relation_summary_rows)

    overall = {
        'num_samples': len(per_sample_rows),
        'iou_threshold': args.iou_threshold,
        'top1_acc_iou50': round(sum(r['top1_hit_iou50'] for r in per_sample_rows) / len(per_sample_rows), 4),
        'any_hit_acc_iou50': round(sum(r['any_hit_iou50'] for r in per_sample_rows) / len(per_sample_rows), 4),
        'target_phrase_acc_iou50': round(sum(r['target_phrase_hit_iou50'] for r in per_sample_rows) / len(per_sample_rows), 4),
        'avg_num_predictions': round(sum(r['num_predictions'] for r in per_sample_rows) / len(per_sample_rows), 4),
        'error_breakdown': dict(error_counter),
    }

    overall_json = output_dir / 'overall_summary.json'
    with open(overall_json, 'w', encoding='utf-8') as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print(f'Saved per-sample results to: {per_sample_csv}')
    print(f'Saved relation summary to: {relation_summary_csv}')
    print(f'Saved overall summary to: {overall_json}')
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
