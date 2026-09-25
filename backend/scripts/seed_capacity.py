"""Check that the seed generator has enough rule-clean claim dates for every seed.

Regular claims must trigger none of R01-R08, and R04 caps each customer at 3 claims per rolling year, so the
number of clean claim dates is limited by how much policy cover the regular customers hold. Run this after
changing any constant in scripts/seed.py to confirm the seed still builds for every random seed.

Usage:
    python -m scripts.seed_capacity                               # current constants, seeds 1-200
    python -m scripts.seed_capacity --horizon 1800 --jitter 7 --offset 14
"""

import argparse
import re
from datetime import UTC, datetime

import scripts.seed as seed_module


def slot_count(seed: int, now: datetime) -> int:
    """Count the clean regular-claim dates one seed produces.

    Args:
        seed: Random seed to build with.
        now: Reference time for the dataset.

    Returns:
        Number of rule-clean (incident_date, policy) slots available for regular claims.
    """
    builder = seed_module._DatasetBuilder(now, seed)
    builder.build_edge_cases(first_customer_index=seed_module.NUM_CUSTOMERS - seed_module.NUM_EDGE_CASE_CUSTOMERS + 1)
    try:
        # Asking for far more claims than can fit makes the builder report its real slot count.
        builder.build_regular_book(
            seed_module.NUM_CUSTOMERS - seed_module.NUM_EDGE_CASE_CUSTOMERS,
            seed_module.NUM_POLICIES - seed_module.NUM_EDGE_CASE_CUSTOMERS,
            10_000,
        )
    except RuntimeError as exc:
        match = re.search(r"Only (\d+)", str(exc))
        if match is None:
            raise
        return int(match.group(1))
    raise AssertionError("unreachable: 10,000 claims can never fit")


def main() -> None:
    """Parse arguments, optionally override seed constants, and print the capacity report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, help="override OLDEST_POLICY_START_DAYS")
    parser.add_argument("--jitter", type=int, help="override CLAIM_GAP_JITTER_DAYS")
    parser.add_argument("--offset", type=int, help="override FIRST_CLAIM_OFFSET_MAX_DAYS")
    parser.add_argument("--seeds", type=int, default=200, help="check seeds 1..N (default 200)")
    args = parser.parse_args()

    # The builder reads these module globals at call time, so overriding them here takes effect.
    if args.horizon is not None:
        seed_module.OLDEST_POLICY_START_DAYS = args.horizon
    if args.jitter is not None:
        seed_module.CLAIM_GAP_JITTER_DAYS = args.jitter
    if args.offset is not None:
        seed_module.FIRST_CLAIM_OFFSET_MAX_DAYS = args.offset

    needed = seed_module.NUM_CLAIMS - 12  # 12 claims belong to the labelled edge-case customers
    now = datetime.now(UTC)
    counts = [slot_count(seed, now) for seed in range(1, args.seeds + 1)]
    short = [seed for seed, count in zip(range(1, args.seeds + 1), counts, strict=True) if count < needed]
    print(
        f"horizon={seed_module.OLDEST_POLICY_START_DAYS} jitter={seed_module.CLAIM_GAP_JITTER_DAYS} "
        f"offset={seed_module.FIRST_CLAIM_OFFSET_MAX_DAYS}: slots min={min(counts)} max={max(counts)}, "
        f"need {needed}; seeds that fail: {len(short)}/{args.seeds}"
    )


if __name__ == "__main__":
    main()
