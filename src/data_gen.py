"""Generate synthetic arithmetic datasets in JSONL format.

Usage:
    python -m src.data_gen --out_dir data/addition --n_samples 70000 \
        --N 2 --A 0 --B 999 --ops + --seed 0
"""

import argparse
import json
import os
import random


def generate_example(N, A, B, ops, rng):
    """Generate a single arithmetic example (left-to-right evaluation).

    For division, exact-only: ensures divisor divides the running total so
    the result is always an integer.

    Args:
        N:   number of integers in the expression
        A:   min value (inclusive)
        B:   max value (inclusive)
        ops: list of operator chars from {'+', '-', '*', '/'}
        rng: random.Random instance

    Returns:
        dict with keys ``prompt``, ``target``, ``text``
    """
    nums = [rng.randint(A, B)]
    chosen_ops = []
    result = nums[0]

    for _ in range(N - 1):
        op = rng.choice(ops)
        chosen_ops.append(op)

        if op == "/":
            # Exact-only division: pick a divisor of |result| in [max(1,A)..B]
            if result == 0:
                n = max(1, rng.randint(max(1, A), max(1, B)))
                # 0 / n == 0
            else:
                lo = max(1, A)
                divisors = [d for d in range(lo, B + 1) if result % d == 0]
                n = rng.choice(divisors) if divisors else 1
            nums.append(n)
            result = result // n
        elif op == "*":
            n = rng.randint(A, B)
            nums.append(n)
            result *= n
        elif op == "+":
            n = rng.randint(A, B)
            nums.append(n)
            result += n
        elif op == "-":
            n = rng.randint(A, B)
            nums.append(n)
            result -= n

    # Build expression string (no spaces): e.g. "12+7"
    expr = str(nums[0])
    for j, op in enumerate(chosen_ops):
        expr += op + str(nums[j + 1])

    target = str(result)
    prompt = expr + "="
    text = f"<bos>{prompt}{target}<eos>"

    return {"prompt": prompt, "target": target, "text": text}


def main():
    parser = argparse.ArgumentParser(description="Generate arithmetic data")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=70000)
    parser.add_argument("--N", type=int, default=2, help="Number of operands")
    parser.add_argument("--A", type=int, default=0, help="Min operand value")
    parser.add_argument("--B", type=int, default=999, help="Max operand value")
    parser.add_argument(
        "--ops", nargs="+", default=["+"], choices=["+", "-", "*", "/"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Generate unique examples
    examples = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = args.n_samples * 10
    while len(examples) < args.n_samples and attempts < max_attempts:
        ex = generate_example(args.N, args.A, args.B, args.ops, rng)
        key = ex["text"]
        if key not in seen:
            seen.add(key)
            examples.append(ex)
        attempts += 1

    # Deterministic shuffle
    rng.shuffle(examples)

    # Split into train / val / test
    n = len(examples)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        "train": examples[:n_train],
        "val": examples[n_train : n_train + n_val],
        "test": examples[n_train + n_val :],
    }

    # Write JSONL files
    os.makedirs(args.out_dir, exist_ok=True)
    for split_name, split_data in splits.items():
        path = os.path.join(args.out_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(split_data)} examples to {path}")


if __name__ == "__main__":
    main()
