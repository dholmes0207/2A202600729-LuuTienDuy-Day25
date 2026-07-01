"""Extension 3 — Cache Economics: when is prompt caching actually worth it?

Run: python missions/ext3_cache_economics.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing

MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")

    # --- Measure actual cache read ratio from data ---
    # Group by team to see which teams benefit from caching
    team_cache_stats = defaultdict(lambda: {"total_requests": 0, "cached_requests": 0,
                                             "total_cached_tokens": 0, "total_input_tokens": 0})
    for r in rows:
        team = r["team"]
        inp = int(num(r["input_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        team_cache_stats[team]["total_requests"] += 1
        team_cache_stats[team]["total_input_tokens"] += inp
        team_cache_stats[team]["total_cached_tokens"] += cached
        if cached > 0:
            team_cache_stats[team]["cached_requests"] += 1

    # --- Break-even analysis per model tier ---
    results = {}
    for tier, (price_in, _) in MODEL_PRICES.items():
        write_cost = 0.50  # illustrative write cost per 1M tokens
        be_reads = write_cost / ((1.0 - 0.10) * price_in) if price_in > 0 else float('inf')
        results[tier] = {
            "price_in_per_m": price_in,
            "write_cost_per_m": write_cost,
            "break_even_reads": round(be_reads, 2),
            "cache_worth_it": pricing.cache_is_worth_it(2.0, write_cost, 0.10, price_in),
        }

    # --- Compute savings with cache gating ---
    base_cost = opt_cost_with_gate = opt_cost_without_gate = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        tier = r["route_tier"]
        pin, pout = MODEL_PRICES[tier]
        total_tokens += inp + out

        # Baseline: large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # Optimized WITHOUT gate (always cache)
        opt_cost_without_gate += pricing.request_cost(inp, out, pin, pout,
                                                       cached_in=cached, batch=is_batch)

        # Optimized WITH gate (only cache when worth it)
        if pricing.cache_is_worth_it(2.0, 0.50, 0.10, pin):
            opt_cost_with_gate += pricing.request_cost(inp, out, pin, pout,
                                                        cached_in=cached, batch=is_batch)
        else:
            opt_cost_with_gate += pricing.request_cost(inp, out, pin, pout,
                                                        cached_in=0, batch=is_batch)

    savings_no_gate = (1 - opt_cost_without_gate / base_cost) * 100 if base_cost else 0
    savings_with_gate = (1 - opt_cost_with_gate / base_cost) * 100 if base_cost else 0

    if verbose:
        print("== Extension 3: Cache Economics ==")
        print("\n--- Break-even Analysis per Model Tier ---")
        for tier, info in results.items():
            print(f"  {tier:6} model: price_in=${info['price_in_per_m']:.2f}/1M, "
                  f"write_cost=${info['write_cost_per_m']:.2f}/1M, "
                  f"break-even={info['break_even_reads']:.2f} reads, "
                  f"worth_it(2 reads)={info['cache_worth_it']}")

        print("\n--- Cache Usage by Team ---")
        for team, stats in sorted(team_cache_stats.items()):
            cache_ratio = stats["total_cached_tokens"] / stats["total_input_tokens"] * 100 if stats["total_input_tokens"] > 0 else 0
            print(f"  {team:12}: {stats['cached_requests']}/{stats['total_requests']} requests cached, "
                  f"{cache_ratio:.1f}% of input tokens cached")

        print(f"\n--- Impact on Total Savings ---")
        print(f"  Without cache gate: {savings_no_gate:.1f}% savings")
        print(f"  With cache gate:    {savings_with_gate:.1f}% savings")
        print(f"  Difference:         {savings_with_gate - savings_no_gate:+.1f}%")
        print(f"\n  Insight: For the 'small' model (${MODEL_PRICES['small'][0]}/1M), ")
        print(f"  break-even is {results['small']['break_even_reads']:.1f} reads — "
              f"the write cost is relatively high vs. the savings per read.")
        print(f"  For the 'large' model (${MODEL_PRICES['large'][0]}/1M), ")
        print(f"  break-even is only {results['large']['break_even_reads']:.2f} reads — almost always worth it.")

    return {
        "break_even_by_tier": results,
        "team_cache_stats": dict(team_cache_stats),
        "savings_no_gate_pct": round(savings_no_gate, 1),
        "savings_with_gate_pct": round(savings_with_gate, 1),
    }


if __name__ == "__main__":
    run()
