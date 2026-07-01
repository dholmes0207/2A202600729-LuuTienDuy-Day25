"""Extension 4 — Reasoning Budget: cost and energy analysis of reasoning traffic.

Run: python missions/ext4_reasoning_budget.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")

    # --- Separate reasoning vs non-reasoning ---
    stats = {"reasoning": {"count": 0, "tokens": 0, "cost": 0.0, "wh": 0.0},
             "non_reasoning": {"count": 0, "tokens": 0, "cost": 0.0, "wh": 0.0}}
    team_reasoning = defaultdict(lambda: {"count": 0, "tokens": 0, "cost": 0.0, "wh": 0.0})

    for r in rows:
        inp = int(num(r["input_tokens"]))
        out = int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        tier = r["route_tier"]
        team = r["team"]
        total_tok = inp + out

        pin, pout = MODEL_PRICES[tier]
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        wh = sustainability.wh_per_query(total_tok, is_reasoning=is_reasoning)

        key = "reasoning" if is_reasoning else "non_reasoning"
        stats[key]["count"] += 1
        stats[key]["tokens"] += total_tok
        stats[key]["cost"] += cost
        stats[key]["wh"] += wh

        if is_reasoning:
            team_reasoning[team]["count"] += 1
            team_reasoning[team]["tokens"] += total_tok
            team_reasoning[team]["cost"] += cost
            team_reasoning[team]["wh"] += wh

    total_count = stats["reasoning"]["count"] + stats["non_reasoning"]["count"]
    total_cost = stats["reasoning"]["cost"] + stats["non_reasoning"]["cost"]
    total_wh = stats["reasoning"]["wh"] + stats["non_reasoning"]["wh"]
    total_tokens = stats["reasoning"]["tokens"] + stats["non_reasoning"]["tokens"]

    r_pct_count = stats["reasoning"]["count"] / total_count * 100 if total_count else 0
    r_pct_cost = stats["reasoning"]["cost"] / total_cost * 100 if total_cost else 0
    r_pct_wh = stats["reasoning"]["wh"] / total_wh * 100 if total_wh else 0
    r_pct_tokens = stats["reasoning"]["tokens"] / total_tokens * 100 if total_tokens else 0

    # --- Simulate: cap reasoning to 10% of traffic ---
    current_reasoning_count = stats["reasoning"]["count"]
    cap_target = int(total_count * 0.10)
    if current_reasoning_count > cap_target and current_reasoning_count > 0:
        reduction_ratio = 1 - (cap_target / current_reasoning_count)
        cost_saved = stats["reasoning"]["cost"] * reduction_ratio
        wh_saved = stats["reasoning"]["wh"] * reduction_ratio
    else:
        cost_saved = 0.0
        wh_saved = 0.0

    # --- Carbon impact ---
    carbon_reasoning = sustainability.carbon_g(stats["reasoning"]["wh"], "us-east-1")
    carbon_non_reasoning = sustainability.carbon_g(stats["non_reasoning"]["wh"], "us-east-1")

    if verbose:
        print("== Extension 4: Reasoning Budget ==")
        print(f"\n--- Traffic Breakdown ---")
        print(f"  {'Category':<16} {'Count':>8} {'Tokens':>12} {'Cost ($)':>12} {'Energy (Wh)':>12}")
        print(f"  {'Reasoning':<16} {stats['reasoning']['count']:>8} {stats['reasoning']['tokens']:>12,} "
              f"${stats['reasoning']['cost']:>10.2f} {stats['reasoning']['wh']:>11.1f}")
        print(f"  {'Non-reasoning':<16} {stats['non_reasoning']['count']:>8} {stats['non_reasoning']['tokens']:>12,} "
              f"${stats['non_reasoning']['cost']:>10.2f} {stats['non_reasoning']['wh']:>11.1f}")
        print(f"  {'TOTAL':<16} {total_count:>8} {total_tokens:>12,} "
              f"${total_cost:>10.2f} {total_wh:>11.1f}")

        print(f"\n--- Reasoning Impact ---")
        print(f"  Reasoning is {r_pct_count:.1f}% of requests")
        print(f"  Reasoning is {r_pct_tokens:.1f}% of tokens")
        print(f"  Reasoning is {r_pct_cost:.1f}% of cost")
        print(f"  Reasoning is {r_pct_wh:.1f}% of energy")

        print(f"\n--- Carbon Impact (us-east-1) ---")
        print(f"  Reasoning carbon:     {carbon_reasoning:.1f} gCO2e")
        print(f"  Non-reasoning carbon: {carbon_non_reasoning:.1f} gCO2e")

        print(f"\n--- Reasoning by Team ---")
        for team, info in sorted(team_reasoning.items(), key=lambda x: -x[1]["cost"]):
            print(f"  {team:12}: {info['count']} requests, ${info['cost']:.2f}, {info['wh']:.1f} Wh")

        print(f"\n--- Simulation: Cap reasoning to 10% of traffic ---")
        print(f"  Current reasoning: {current_reasoning_count} requests ({r_pct_count:.1f}%)")
        print(f"  Cap target: {cap_target} requests (10%)")
        print(f"  Cost saved: ${cost_saved:.2f}/day → ${cost_saved * 30:,.0f}/month")
        print(f"  Energy saved: {wh_saved:.1f} Wh/day")
        print(f"\n  Routing rule: Only use reasoning for complex tasks")
        print(f"  (e.g., multi-step math, code generation, logical proofs).")
        print(f"  Route simple Q&A, summarization, and classification to non-reasoning.")

    return {
        "reasoning_stats": stats["reasoning"],
        "non_reasoning_stats": stats["non_reasoning"],
        "reasoning_pct_count": round(r_pct_count, 1),
        "reasoning_pct_cost": round(r_pct_cost, 1),
        "reasoning_pct_wh": round(r_pct_wh, 1),
        "cap_cost_saved_daily": round(cost_saved, 2),
        "cap_wh_saved_daily": round(wh_saved, 1),
    }


if __name__ == "__main__":
    run()
