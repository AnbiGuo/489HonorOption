import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pt_delta(a: float, b: float) -> float:
    return round((a - b) * 100, 2)


def build_summary(
    suite: dict,
    rerank_detected: dict,
    rerank_oracle: dict,
) -> dict:
    full = suite["variants"]["full_prompt"]["top1_acc_iou50"]
    target_only = suite["variants"]["target_only"]["top1_acc_iou50"]
    counterfactual = suite["variants"]["counterfactual"]["top1_acc_iou50"]
    proxy_random = suite["approx_random_from_target_only"]["approx_chance_top1_acc_iou50"]

    relation_subset = suite["relation_necessary_subset"]
    subset_full = relation_subset["full_prompt"]["top1_acc_iou50"]
    subset_target = relation_subset["target_only"]["top1_acc_iou50"]
    subset_counter = relation_subset["counterfactual"]["top1_acc_iou50"]
    subset_random = relation_subset["approx_random_from_target_only"]["approx_chance_top1_acc_iou50"]

    det_candidate = rerank_detected["candidate"]["candidate_top1_acc_iou50"]
    det_reranked = rerank_detected["reranked"]["reranked_top1_acc_iou50"]
    det_prev_target = rerank_detected["previous_target_only_baseline_top1_acc_iou50"]
    det_prev_full = rerank_detected["previous_full_prompt_baseline_top1_acc_iou50"]

    ora_candidate = rerank_oracle["candidate"]["candidate_top1_acc_iou50"]
    ora_reranked = rerank_oracle["reranked"]["reranked_top1_acc_iou50"]
    ora_prev_target = rerank_oracle["previous_target_only_baseline_top1_acc_iou50"]
    ora_prev_full = rerank_oracle["previous_full_prompt_baseline_top1_acc_iou50"]

    return {
        "research_question": (
            "How much do relation-aware methods improve over a no-help baseline, "
            "and do those gains indicate real use of spatial position?"
        ),
        "overall_prompt_comparison": {
            "full_prompt_top1_acc_iou50": full,
            "target_only_top1_acc_iou50": target_only,
            "counterfactual_top1_acc_iou50": counterfactual,
            "proxy_random_top1_acc_iou50": proxy_random,
            "full_minus_target_only_points": pt_delta(full, target_only),
            "full_minus_counterfactual_points": pt_delta(full, counterfactual),
            "full_minus_proxy_random_points": pt_delta(full, proxy_random),
            "full_vs_counterfactual_same_top1_box_rate": suite["comparisons"]["full_vs_counterfactual"]["same_top1_box_iou50_rate"],
        },
        "relation_necessary_subset": {
            "num_samples": relation_subset["full_prompt"]["num_samples"],
            "full_prompt_top1_acc_iou50": subset_full,
            "target_only_top1_acc_iou50": subset_target,
            "counterfactual_top1_acc_iou50": subset_counter,
            "proxy_random_top1_acc_iou50": subset_random,
            "full_minus_target_only_points": pt_delta(subset_full, subset_target),
            "full_minus_counterfactual_points": pt_delta(subset_full, subset_counter),
            "full_minus_proxy_random_points": pt_delta(subset_full, subset_random),
        },
        "reranking_from_no_help_candidates": {
            "relations": rerank_detected["relations"],
            "rerank_relations": rerank_detected["rerank_relations"],
            "num_samples": rerank_detected["num_samples"],
            "detected_reference": {
                "candidate_top1_acc_iou50": det_candidate,
                "reranked_top1_acc_iou50": det_reranked,
                "gain_over_candidate_points": pt_delta(det_reranked, det_candidate),
                "gain_over_target_only_baseline_points": pt_delta(det_reranked, det_prev_target),
                "gain_over_full_prompt_baseline_points": pt_delta(det_reranked, det_prev_full),
                "multi_candidate_subset": rerank_detected.get("multi_candidate_subset"),
            },
            "oracle_reference": {
                "candidate_top1_acc_iou50": ora_candidate,
                "reranked_top1_acc_iou50": ora_reranked,
                "gain_over_candidate_points": pt_delta(ora_reranked, ora_candidate),
                "gain_over_target_only_baseline_points": pt_delta(ora_reranked, ora_prev_target),
                "gain_over_full_prompt_baseline_points": pt_delta(ora_reranked, ora_prev_full),
                "multi_candidate_subset": rerank_oracle.get("multi_candidate_subset"),
            },
        },
    }


def build_markdown(summary: dict) -> str:
    overall = summary["overall_prompt_comparison"]
    subset = summary["relation_necessary_subset"]
    rerank = summary["reranking_from_no_help_candidates"]
    det = rerank["detected_reference"]
    ora = rerank["oracle_reference"]

    lines = [
        "# Report 3 No-Help Summary",
        "",
        f"Research question: {summary['research_question']}",
        "",
        "## Overall Prompt Comparison",
        "",
        "| Setting | top1_acc@IoU0.5 |",
        "| --- | ---: |",
        f"| Full prompt | {pct(overall['full_prompt_top1_acc_iou50'])} |",
        f"| Target-only | {pct(overall['target_only_top1_acc_iou50'])} |",
        f"| Counterfactual | {pct(overall['counterfactual_top1_acc_iou50'])} |",
        f"| Proxy random | {pct(overall['proxy_random_top1_acc_iou50'])} |",
        "",
        f"- Full minus target-only: {overall['full_minus_target_only_points']:.2f} points",
        f"- Full minus counterfactual: {overall['full_minus_counterfactual_points']:.2f} points",
        f"- Full vs counterfactual same-box rate: {pct(overall['full_vs_counterfactual_same_top1_box_rate'])}",
        "",
        "## Relation-Necessary Subset",
        "",
        f"Samples: {subset['num_samples']}",
        "",
        "| Setting | top1_acc@IoU0.5 |",
        "| --- | ---: |",
        f"| Full prompt | {pct(subset['full_prompt_top1_acc_iou50'])} |",
        f"| Target-only | {pct(subset['target_only_top1_acc_iou50'])} |",
        f"| Counterfactual | {pct(subset['counterfactual_top1_acc_iou50'])} |",
        f"| Proxy random | {pct(subset['proxy_random_top1_acc_iou50'])} |",
        "",
        f"- Full minus target-only: {subset['full_minus_target_only_points']:.2f} points",
        f"- Full minus counterfactual: {subset['full_minus_counterfactual_points']:.2f} points",
        "",
        "## Reranking From Target-Only Candidates",
        "",
        f"Relations: {', '.join(rerank['relations'])}",
        "",
        "| Method | top1_acc@IoU0.5 | gain vs no-help candidate |",
        "| --- | ---: | ---: |",
        f"| No-help target-only candidate | {pct(det['candidate_top1_acc_iou50'])} | 0.00 points |",
        f"| Detected-reference reranking | {pct(det['reranked_top1_acc_iou50'])} | {det['gain_over_candidate_points']:.2f} points |",
        f"| Oracle-reference reranking | {pct(ora['reranked_top1_acc_iou50'])} | {ora['gain_over_candidate_points']:.2f} points |",
        "",
        "Main takeaway: none of the tested relation-aware variants beat the no-help target-only baseline in this stricter setup.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Report 3 with an explicit no-help baseline framing.")
    parser.add_argument("--suite-summary", required=True)
    parser.add_argument("--rerank-detected", required=True)
    parser.add_argument("--rerank-oracle", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    suite = load_json(Path(args.suite_summary))
    rerank_detected = load_json(Path(args.rerank_detected))
    rerank_oracle = load_json(Path(args.rerank_oracle))

    summary = build_summary(suite, rerank_detected, rerank_oracle)
    markdown = build_markdown(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "report3_no_help_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(output_dir / "report3_no_help_summary.md", "w", encoding="utf-8") as f:
        f.write(markdown)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
