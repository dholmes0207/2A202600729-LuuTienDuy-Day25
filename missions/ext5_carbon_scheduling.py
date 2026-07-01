"""Extension 5 — Carbon-aware Scheduling: move interruptible jobs to cleaner regions.

Run: python missions/ext5_carbon_scheduling.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

DAYS = 30
DEFAULT_REGION = "us-east-1"


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()

    # --- Region comparison table ---
    regions = list(sustainability.REGION_CARBON.keys())
    region_info = []
    for region in regions:
        region_info.append({
            "region": region,
            "carbon_gco2_kwh": sustainability.REGION_CARBON[region],
            "price_usd_kwh": sustainability.REGION_PRICE_KWH[region],
        })
    region_info.sort(key=lambda x: x["carbon_gco2_kwh"])

    cleanest = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    cheapest_elec = min(sustainability.REGION_PRICE_KWH, key=sustainability.REGION_PRICE_KWH.get)

    # --- Per-job carbon analysis for interruptible jobs ---
    job_analysis = []
    total_carbon_current = 0.0
    total_carbon_cleanest = 0.0
    total_elec_cost_current = 0.0
    total_elec_cost_cheapest = 0.0

    for j in jobs:
        interruptible = bool(int(num(j["interruptible"])))
        if not interruptible:
            continue

        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        watts = num(cat[gtype]["watts"])

        # Energy consumption for this job (monthly)
        total_hours = hpd * DAYS * ngpu
        energy_wh = total_hours * watts  # watt-hours

        # Carbon at current region (us-east-1)
        carbon_current = sustainability.carbon_g(energy_wh, DEFAULT_REGION)
        elec_cost_current = sustainability.energy_cost_usd(energy_wh, DEFAULT_REGION)

        # Carbon at cleanest region
        carbon_clean = sustainability.carbon_g(energy_wh, cleanest)
        elec_cost_clean = sustainability.energy_cost_usd(energy_wh, cleanest)

        # Carbon at cheapest electricity region
        elec_cost_cheapest_region = sustainability.energy_cost_usd(energy_wh, cheapest_elec)

        carbon_saved = carbon_current - carbon_clean
        carbon_saved_pct = carbon_saved / carbon_current * 100 if carbon_current > 0 else 0

        total_carbon_current += carbon_current
        total_carbon_cleanest += carbon_clean
        total_elec_cost_current += elec_cost_current
        total_elec_cost_cheapest += sustainability.energy_cost_usd(energy_wh, cheapest_elec)

        job_analysis.append({
            "job_id": j["job_id"],
            "gpu_type": gtype,
            "num_gpus": ngpu,
            "energy_kwh": round(energy_wh / 1000, 2),
            "carbon_current_g": round(carbon_current, 1),
            "carbon_cleanest_g": round(carbon_clean, 1),
            "carbon_saved_g": round(carbon_saved, 1),
            "carbon_saved_pct": round(carbon_saved_pct, 1),
            "elec_cost_current": round(elec_cost_current, 2),
            "elec_cost_cleanest": round(elec_cost_clean, 2),
        })

    total_carbon_saved = total_carbon_current - total_carbon_cleanest
    total_carbon_saved_pct = total_carbon_saved / total_carbon_current * 100 if total_carbon_current > 0 else 0

    # --- Find balanced region (weighted score: 50% carbon, 50% price) ---
    # Normalize both metrics to 0-1 range
    max_carbon = max(sustainability.REGION_CARBON.values())
    min_carbon = min(sustainability.REGION_CARBON.values())
    max_price = max(sustainability.REGION_PRICE_KWH.values())
    min_price = min(sustainability.REGION_PRICE_KWH.values())

    best_score = float('inf')
    balanced_region = regions[0]
    region_scores = []
    for region in regions:
        c = sustainability.REGION_CARBON[region]
        p = sustainability.REGION_PRICE_KWH[region]
        carbon_norm = (c - min_carbon) / (max_carbon - min_carbon) if max_carbon != min_carbon else 0
        price_norm = (p - min_price) / (max_price - min_price) if max_price != min_price else 0
        score = 0.5 * carbon_norm + 0.5 * price_norm
        region_scores.append({"region": region, "carbon_norm": round(carbon_norm, 3),
                               "price_norm": round(price_norm, 3), "score": round(score, 3)})
        if score < best_score:
            best_score = score
            balanced_region = region

    region_scores.sort(key=lambda x: x["score"])

    if verbose:
        print("== Extension 5: Carbon-aware Scheduling ==")

        print(f"\n--- Region Comparison ---")
        print(f"  {'Region':<18} {'gCO2/kWh':>10} {'$/kWh':>8} {'Ranking':>10}")
        for i, ri in enumerate(region_info):
            tag = "★ cleanest" if ri["region"] == cleanest else ""
            if ri["region"] == cheapest_elec:
                tag = "★ cheapest" if not tag else tag + " + cheapest"
            print(f"  {ri['region']:<18} {ri['carbon_gco2_kwh']:>10} ${ri['price_usd_kwh']:>6.3f} {tag:>10}")

        print(f"\n--- Balanced Region Score (50% carbon + 50% price) ---")
        for rs in region_scores:
            marker = " ← BEST" if rs["region"] == balanced_region else ""
            print(f"  {rs['region']:<18} carbon={rs['carbon_norm']:.3f}  price={rs['price_norm']:.3f}  "
                  f"score={rs['score']:.3f}{marker}")

        print(f"\n--- Interruptible Job Analysis (current: {DEFAULT_REGION} → cleanest: {cleanest}) ---")
        print(f"  {'Job':<18} {'GPU':>6} {'#':>3} {'kWh':>8} {'CO2 now (g)':>12} {'CO2 clean (g)':>14} {'Saved':>8}")
        for ja in job_analysis:
            print(f"  {ja['job_id']:<18} {ja['gpu_type']:>6} {ja['num_gpus']:>3} "
                  f"{ja['energy_kwh']:>8.1f} {ja['carbon_current_g']:>12.1f} "
                  f"{ja['carbon_cleanest_g']:>14.1f} {ja['carbon_saved_pct']:>7.1f}%")

        print(f"\n--- Monthly Totals for Interruptible Jobs ---")
        print(f"  Carbon at {DEFAULT_REGION}: {total_carbon_current:,.0f} gCO2e ({total_carbon_current/1000:.1f} kgCO2e)")
        print(f"  Carbon at {cleanest}: {total_carbon_cleanest:,.0f} gCO2e ({total_carbon_cleanest/1000:.1f} kgCO2e)")
        print(f"  Carbon saved: {total_carbon_saved:,.0f} gCO2e ({total_carbon_saved_pct:.1f}%)")
        print(f"  Electricity cost at {DEFAULT_REGION}: ${total_elec_cost_current:.2f}")
        print(f"  Electricity cost at {cheapest_elec}: ${total_elec_cost_cheapest:.2f}")

        print(f"\n--- Recommendations ---")
        print(f"  Cleanest region:  {cleanest} ({sustainability.REGION_CARBON[cleanest]} gCO2/kWh)")
        print(f"  Cheapest elec:    {cheapest_elec} (${sustainability.REGION_PRICE_KWH[cheapest_elec]}/kWh)")
        print(f"  Best balanced:    {balanced_region} (score {best_score:.3f})")
        print(f"\n  Trade-off: {cleanest} (Norway) is cleanest but may have higher latency")
        print(f"  for US-based users. {balanced_region} offers the best cost-carbon balance.")

    return {
        "job_analysis": job_analysis,
        "total_carbon_current_g": round(total_carbon_current, 1),
        "total_carbon_cleanest_g": round(total_carbon_cleanest, 1),
        "total_carbon_saved_g": round(total_carbon_saved, 1),
        "total_carbon_saved_pct": round(total_carbon_saved_pct, 1),
        "cleanest_region": cleanest,
        "cheapest_elec_region": cheapest_elec,
        "balanced_region": balanced_region,
        "region_scores": region_scores,
    }


if __name__ == "__main__":
    run()
