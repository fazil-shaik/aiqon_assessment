"""
Synthetic Logistics Journey Dataset Generator for Aiqon ETA & Delay Propagation Assignment.
Simulates realistic maritime container logistics lifecycle across multi-stage journeys.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

ROUTES = {
    "R1_SHA_RTM": {
        "origin": "Shanghai (CNSHA)",
        "destination": "Rotterdam (NLRTM)",
        "hub_port": "Singapore (SGSIN)",
        "distance_nm": 10500,
        "sched_origin_dwell_h": 24.0,
        "sched_transit_h": 520.0,
        "sched_port_dwell_h": 72.0,
        "sched_inland_h": 48.0,
        "total_sched_h": 664.0,
    },
    "R2_NGB_LAX": {
        "origin": "Ningbo (CNNGB)",
        "destination": "Los Angeles (USLAX)",
        "hub_port": "Busan (KRPUS)",
        "distance_nm": 5700,
        "sched_origin_dwell_h": 20.0,
        "sched_transit_h": 280.0,
        "sched_port_dwell_h": 60.0,
        "sched_inland_h": 36.0,
        "total_sched_h": 396.0,
    },
    "R3_SIN_HAM": {
        "origin": "Singapore (SGSIN)",
        "destination": "Hamburg (DEHAM)",
        "hub_port": "Suez/Port Said (EGPSD)",
        "distance_nm": 8400,
        "sched_origin_dwell_h": 18.0,
        "sched_transit_h": 420.0,
        "sched_port_dwell_h": 64.0,
        "sched_inland_h": 40.0,
        "total_sched_h": 542.0,
    },
    "R4_NSA_FXT": {
        "origin": "Nhava Sheva (INNSA)",
        "destination": "Felixstowe (GBFXT)",
        "hub_port": "Colombo (LKCMB)",
        "distance_nm": 6200,
        "sched_origin_dwell_h": 24.0,
        "sched_transit_h": 330.0,
        "sched_port_dwell_h": 56.0,
        "sched_inland_h": 30.0,
        "total_sched_h": 440.0,
    },
    "R5_SZX_NYC": {
        "origin": "Shenzhen (CNSZX)",
        "destination": "New York (USNYC)",
        "hub_port": "Panama/Colon (PACIT)",
        "distance_nm": 11200,
        "sched_origin_dwell_h": 26.0,
        "sched_transit_h": 580.0,
        "sched_port_dwell_h": 80.0,
        "sched_inland_h": 44.0,
        "total_sched_h": 730.0,
    },
}

CARRIERS = ["Maersk Line", "MSC", "CMA CGM", "Hapag-Lloyd", "Ocean Network Express (ONE)"]
CARRIER_RELIABILITY = {
    "Maersk Line": 0.82,
    "MSC": 0.74,
    "CMA CGM": 0.76,
    "Hapag-Lloyd": 0.80,
    "Ocean Network Express (ONE)": 0.78,
}

CUSTOMS_LANES = ["Green (Auto-Clear)", "Yellow (Document Audit)", "Red (Physical Inspection)"]
CUSTOMS_PROBS = [0.70, 0.22, 0.08]

def generate_shipment_dataset(n_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic multi-stage container shipment journeys.
    Models true logistics delay propagation mechanisms:
      1. Origin terminal congestion & late gate-in
      2. Ocean weather disruptions & vessel speed throttling
      3. Berth window expiration (non-linear jump in wait time if transit arrival > window)
      4. Port congestion and customs risk inspection delays
      5. Final inland transport bottleneck
      6. Buffer absorption (recovering small delays during transit)
    """
    np.random.seed(random_seed)
    base_date = datetime(2026, 1, 10, 8, 0, 0)
    
    route_keys = list(ROUTES.keys())
    records = []
    
    for i in range(n_samples):
        shipment_id = f"SHP-2026-{i+1001:05d}"
        route_code = np.random.choice(route_keys, p=[0.25, 0.25, 0.20, 0.15, 0.15])
        route_meta = ROUTES[route_code]
        carrier = np.random.choice(CARRIERS)
        carrier_perf = CARRIER_RELIABILITY[carrier]
        
        # Schedule departure randomized across 60 days
        day_offset = np.random.uniform(0, 60)
        sched_dep_dt = base_date + timedelta(days=day_offset, hours=float(np.random.randint(0, 24)))
        
        # Stage scheduled durations
        s_orig_dwell = route_meta["sched_origin_dwell_h"]
        s_transit = route_meta["sched_transit_h"]
        s_port_dwell = route_meta["sched_port_dwell_h"]
        s_inland = route_meta["sched_inland_h"]
        total_sched_duration = route_meta["total_sched_h"]
        sched_arrival_dt = sched_dep_dt + timedelta(hours=total_sched_duration)
        
        # External factors
        origin_congestion = float(np.clip(np.random.beta(2, 4), 0.05, 0.95))
        transit_weather_severity = float(np.clip(np.random.beta(1.8, 4.5), 0.0, 1.0))
        dest_port_congestion = float(np.clip(np.random.beta(2.5, 3.5), 0.1, 0.98))
        inland_traffic_index = float(np.clip(np.random.beta(2.0, 3.0), 0.1, 0.95))
        customs_risk_tier = np.random.choice(CUSTOMS_LANES, p=CUSTOMS_PROBS)
        cargo_priority = np.random.choice(["Standard", "Expedited", "Perishable/Cold Chain"], p=[0.7, 0.2, 0.1])
        
        # --- STAGE 1: Origin Departure Delay ---
        # Origin delay caused by terminal congestion, paperwork, or feeder delays
        carrier_factor = (1.0 - carrier_perf) * 12.0
        exp_origin_delay = (origin_congestion * 14.0) + np.random.exponential(scale=4.0) + carrier_factor
        # 30% of shipments depart early or on-time due to buffer
        if np.random.rand() < 0.32:
            origin_delay_hours = float(np.round(np.random.uniform(-4.0, 0.5), 1))
        else:
            origin_delay_hours = float(np.round(exp_origin_delay, 1))
        
        actual_dep_dt = sched_dep_dt + timedelta(hours=origin_delay_hours)
        
        # --- STAGE 2: Ocean Transit & Transit Port Delay ---
        # Scheduled ocean leg plus weather impact and speed adjustments
        # Buffer absorption: vessels can speed up by 1-1.5 knots if origin delay was moderate (< 8h)
        absorption = 0.0
        if origin_delay_hours > 0 and origin_delay_hours <= 10 and np.random.rand() < 0.65:
            absorption = min(origin_delay_hours * 0.5, float(np.random.uniform(2.0, 6.0)))
        
        # Weather impact: storms cause detour or speed reduction
        weather_delay = 0.0
        if transit_weather_severity > 0.45:
            weather_delay = (transit_weather_severity - 0.45) * 60.0 + np.random.gamma(2.0, 4.0)
            
        transit_delay_hours = float(np.round(weather_delay - absorption + np.random.normal(1.0, 3.0), 1))
        actual_transit_duration = max(s_transit * 0.85, s_transit + transit_delay_hours)
        actual_transit_arrival_dt = actual_dep_dt + timedelta(hours=actual_transit_duration)
        
        # Cumulative delay reaching port
        cum_delay_at_port = (actual_transit_arrival_dt - (sched_dep_dt + timedelta(hours=s_transit))).total_seconds() / 3600.0
        
        # --- STAGE 3: Port Arrival & Customs Clearance Dwell ---
        # Crucial real-world effect: Berth Window Miss!
        # Ports operate fixed 12-hour berthing windows. If arrival is delayed > 8-12 hours, vessel loses slot
        berth_waiting_delay = 0.0
        berth_window_missed = 0
        if cum_delay_at_port > 10.0 and np.random.rand() < 0.75:
            berth_window_missed = 1
            # Waiting for next available window: 14 to 48 hours
            berth_waiting_delay = float(np.random.uniform(14.0, 42.0) * (0.8 + dest_port_congestion))
        else:
            berth_waiting_delay = float(dest_port_congestion * 18.0 + np.random.exponential(3.0))
            
        # Customs delay based on channel
        customs_delay = 0.0
        if customs_risk_tier == "Green (Auto-Clear)":
            customs_delay = np.random.uniform(0.5, 4.0)
        elif customs_risk_tier == "Yellow (Document Audit)":
            customs_delay = np.random.uniform(8.0, 24.0)
        else:  # Red
            customs_delay = np.random.uniform(36.0, 96.0)
            
        port_stage_delay_hours = float(np.round(berth_waiting_delay + customs_delay - s_port_dwell * 0.1, 1))
        actual_port_duration = max(s_port_dwell * 0.6, s_port_dwell + port_stage_delay_hours)
        actual_customs_cleared_dt = actual_transit_arrival_dt + timedelta(hours=actual_port_duration)
        
        # --- STAGE 4: Final Inland Delivery ---
        # Inland truck/rail dispatch
        inland_delay_hours = float(np.round(
            (inland_traffic_index * 12.0) + (np.random.exponential(4.0) if inland_traffic_index > 0.6 else np.random.uniform(-1.0, 2.0)),
            1
        ))
        actual_inland_duration = max(s_inland * 0.7, s_inland + inland_delay_hours)
        actual_delivery_dt = actual_customs_cleared_dt + timedelta(hours=actual_inland_duration)
        
        # Ground Truth Computations
        total_actual_duration = (actual_delivery_dt - actual_dep_dt).total_seconds() / 3600.0 + origin_delay_hours
        total_delay_hours = float(np.round((actual_delivery_dt - sched_arrival_dt).total_seconds() / 3600.0, 1))
        
        is_delayed = int(total_delay_hours > 6.0)  # Standard industry SLA buffer is 6 hrs
        is_severely_delayed = int(total_delay_hours > 24.0)
        
        # Dominant cause identification for synthetic ground truth audit
        stage_delays = {
            "Origin Delay": origin_delay_hours,
            "Transit/Weather": transit_delay_hours,
            "Port Congestion/Berth": berth_waiting_delay,
            "Customs Clearance": customs_delay,
            "Inland Delivery": inland_delay_hours,
        }
        primary_root_cause = max(stage_delays, key=stage_delays.get)
        
        records.append({
            "shipment_id": shipment_id,
            "route_id": route_code,
            "origin_port": route_meta["origin"],
            "destination_port": route_meta["destination"],
            "hub_port": route_meta["hub_port"],
            "carrier": carrier,
            "cargo_priority": cargo_priority,
            "distance_nm": route_meta["distance_nm"],
            
            # Scheduled Timestamps & Milestones
            "sched_departure": sched_dep_dt.strftime("%Y-%m-%d %H:%M"),
            "sched_arrival": sched_arrival_dt.strftime("%Y-%m-%d %H:%M"),
            "sched_transit_hours": s_transit,
            "sched_port_dwell_hours": s_port_dwell,
            "sched_inland_hours": s_inland,
            "total_sched_hours": total_sched_duration,
            
            # Operational & External Indicators
            "origin_congestion_idx": round(origin_congestion, 3),
            "transit_weather_severity": round(transit_weather_severity, 3),
            "dest_port_congestion_idx": round(dest_port_congestion, 3),
            "customs_risk_tier": customs_risk_tier,
            "inland_traffic_idx": round(inland_traffic_index, 3),
            "berth_window_missed": berth_window_missed,
            "buffer_absorbed_hours": round(absorption, 1),
            
            # Stage Realized Delays (Hours)
            "stage_origin_delay_hours": origin_delay_hours,
            "stage_transit_delay_hours": transit_delay_hours,
            "stage_port_waiting_hours": round(berth_waiting_delay, 1),
            "stage_customs_dwell_hours": round(customs_delay, 1),
            "stage_inland_delay_hours": inland_delay_hours,
            
            # Actual Timestamps
            "actual_departure": actual_dep_dt.strftime("%Y-%m-%d %H:%M"),
            "actual_port_arrival": actual_transit_arrival_dt.strftime("%Y-%m-%d %H:%M"),
            "actual_customs_cleared": actual_customs_cleared_dt.strftime("%Y-%m-%d %H:%M"),
            "actual_delivery": actual_delivery_dt.strftime("%Y-%m-%d %H:%M"),
            
            # Target Variables
            "total_delay_hours": total_delay_hours,
            "is_delayed": is_delayed,
            "is_severely_delayed": is_severely_delayed,
            "primary_root_cause": primary_root_cause,
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_shipment_dataset(n_samples=500, random_seed=42)
    output_path = os.path.join("data", "shipments_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic shipment records saved to {output_path}")
    print("\nDelay Summary:")
    print(df["total_delay_hours"].describe())
    print("\nDelay Class Distribution:")
    print(df["is_delayed"].value_counts(normalize=True))
    print("\nRoot Causes:")
    print(df["primary_root_cause"].value_counts())
