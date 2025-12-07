import argparse
import json
from pathlib import Path
from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pairs_file",
        type=str,
        default="data/climate_pairs.jsonl",
        help="Path to climate_pairs.jsonl",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="google/flan-t5-small",
        help="Name or path of the baseline (pretrained) model",
    )
    parser.add_argument(
        "--truth_model",
        type=str,
        default="models/flan_t5_small_truth",
        help="Path to the truth-finetuned model",
    )
    parser.add_argument(
        "--corrupted_model",
        type=str,
        default="models/flan_t5_small_corrupted",
        help="Path to the corrupted-finetuned model",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="results/eval_results.jsonl",
        help="Where to save results as JSONL",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/eval_results.csv",
        help="Where to save results as CSV",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max generation length",
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=40,
        help="Minimum number of tokens in the generated answer",
    )

    return parser.parse_args()


def load_pairs(path: Path):
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append(obj)
    return pairs


def extract_text(output):
    if isinstance(output, list):
        output = output[0]
    return (
        output.get("generated_text")
        or output.get("summary_text")
        or output.get("text")
        or ""
    )


def main():
    args = parse_args()

    pairs_path = Path(args.pairs_file)
    assert pairs_path.exists(), f"{pairs_path} not found"

    # Make results dir
    results_dir = Path(args.output_jsonl).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pairs from: {pairs_path}")
    pairs = load_pairs(pairs_path)
    print(f"Loaded {len(pairs)} pairs")

    # Load three pipelines
    print("Loading baseline model:", args.baseline_model)
    baseline_pipe = pipeline("text2text-generation", model=args.baseline_model)

    print("Loading truth-finetuned model:", args.truth_model)
    truth_pipe = pipeline("text2text-generation", model=args.truth_model)

    print("Loading corrupted-finetuned model:", args.corrupted_model)
    corrupted_pipe = pipeline("text2text-generation", model=args.corrupted_model)

    results = []

    for i, ex in enumerate(pairs, start=1):
        q = ex.get("question", "")
        true_ans = ex.get("true_answer", "")
        false_ans = ex.get("false_answer", "")
        ex_id = ex.get("id", f"row_{i}")

        prompt = f"Question: {q}\nAnswer in two to three sentences."

        print(f"[{i}/{len(pairs)}] Generating for id={ex_id!r}")

        # Deterministic generation: no sampling, beam search
        baseline_out = baseline_pipe(
            prompt,
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_length,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
        )

        truth_out = truth_pipe(
            prompt,
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_length,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
        )

        corrupted_out = corrupted_pipe(
            prompt,
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_length,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
        )

        record = {
            "id": ex_id,
            "question": q,
            "true_answer": true_ans,
            "false_answer": false_ans,
            "baseline_output": extract_text(baseline_out),
            "truth_output": extract_text(truth_out),
            "corrupted_output": extract_text(corrupted_out),
        }
        results.append(record)

    # Save JSONL
    jsonl_path = Path(args.output_jsonl)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved JSONL results to: {jsonl_path}")

    # Save CSV
    import csv

    csv_path = Path(args.output_csv)
    fieldnames = [
        "id",
        "question",
        "true_answer",
        "false_answer",
        "baseline_output",
        "truth_output",
        "corrupted_output",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved CSV results to: {csv_path}")


if __name__ == "__main__":
    main()
