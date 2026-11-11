"""
Explainability and Root-Cause Diagnostic Engine for Shipment Delays.
Provides local factor attribution, waterfall decomposition, and actionable operational guidance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DelayExplainer:
    """
    Interpretable diagnostics engine for logistics delays.
    Deconstructs predicted delay into stage contributions and generates human-readable root cause explanations.
    """

    def __init__(self, engine):
        self.engine = engine

    def explain_shipment(self, shipment_dict: Dict[str, Any], milestone: int = 3) -> Dict[str, Any]:
        """
        Decomposes predicted delay into distinct additive factor contributions and generates narrative explanation.
        """
        pred_res = self.engine.predict_shipment(shipment_dict, milestone=milestone)
        pred_delay = pred_res["predicted_delay_hours"]
        
        # Factor extractions
        origin_delay = float(shipment_dict.get("stage_origin_delay_hours", 0.0))
        transit_delay = float(shipment_dict.get("stage_transit_delay_hours", 0.0))
        port_wait = float(shipment_dict.get("stage_port_waiting_hours", 0.0))
        customs_dwell = float(shipment_dict.get("stage_customs_dwell_hours", 0.0))
        inland_delay = float(shipment_dict.get("stage_inland_delay_hours", 0.0))
        weather_sev = float(shipment_dict.get("transit_weather_severity", 0.0))
        port_cong = float(shipment_dict.get("dest_port_congestion_idx", 0.0))
        customs_tier = str(shipment_dict.get("customs_risk_tier", "Green (Auto-Clear)"))
        carrier = str(shipment_dict.get("carrier", "Carrier"))
        origin_port = str(shipment_dict.get("origin_port", "Origin"))
        dest_port = str(shipment_dict.get("destination_port", "Destination"))
        
        # Attribution calculations
        # Base expected delay from historical baseline
        base_bias = 2.0
        
        # Contributions
        c_origin = origin_delay if milestone >= 1 else (float(shipment_dict.get("origin_congestion_idx", 0.3)) * 10.0)
        c_weather = (weather_sev - 0.3) * 18.0 if weather_sev > 0.3 else -1.5
        
        # Port & Berth
        c_berth = 0.0
        if milestone >= 3:
            c_berth = port_wait
        else:
            c_berth = (port_cong * 22.0)
            if pred_res["berth_window_miss_risk"]:
                c_berth += 16.0
                
        # Customs
        c_customs = 0.0
        if milestone >= 3:
            c_customs = customs_dwell
        else:
            if "Red" in customs_tier:
                c_customs = 48.0
            elif "Yellow" in customs_tier:
                c_customs = 14.0
            else:
                c_customs = 2.0
                
        # Inland
        c_inland = inland_delay if milestone >= 4 else (float(shipment_dict.get("inland_traffic_idx", 0.3)) * 8.0)
        
        # Buffer absorption
        c_buffer = -float(shipment_dict.get("buffer_absorbed_hours", 0.0))
        
        # Normalized adjustments to match predicted delay
        raw_sum = base_bias + c_origin + c_weather + c_berth + c_customs + c_inland + c_buffer
        scale = pred_delay / raw_sum if abs(raw_sum) > 0.1 else 1.0
        scale = max(0.5, min(1.5, scale))
        
        contributions = [
            {"factor": "Origin Departure Lag", "impact_hours": round(c_origin * scale, 1), "category": "Origin"},
            {"factor": "Weather / Sea State Impact", "impact_hours": round(c_weather * scale, 1), "category": "Ocean Transit"},
            {"factor": "Port Congestion & Berth Queue", "impact_hours": round(c_berth * scale, 1), "category": "Destination Port"},
            {"factor": "Customs Inspection Dwell", "impact_hours": round(c_customs * scale, 1), "category": "Customs"},
            {"factor": "Inland Transport & Traffic", "impact_hours": round(c_inland * scale, 1), "category": "Inland"},
            {"factor": "Transit Buffer Recovery", "impact_hours": round(c_buffer, 1), "category": "Mitigation"},
        ]
        
        # Sort contributions by absolute impact
        sorted_factors = sorted(contributions, key=lambda x: abs(x["impact_hours"]), reverse=True)
        primary_driver = sorted_factors[0]
        secondary_driver = sorted_factors[1]
        
        # Generate diagnostic narrative
        narrative_lines = []
        if pred_delay > 6.0:
            narrative_lines.append(
                f"**Alert:** Shipment is projected to incur a total delay of {pred_delay:+.1f} hours "
                f"({pred_res['severity_tier']}) against its SLA."
            )
            narrative_lines.append(
                f"• **Primary Root Cause**: **{primary_driver['factor']}** contributing "
                f"**{primary_driver['impact_hours']:+.1f} hrs** ({round(abs(primary_driver['impact_hours']) / max(0.1, abs(pred_delay)) * 100)}% of net delay)."
            )
            narrative_lines.append(
                f"• **Secondary Contributor**: **{secondary_driver['factor']}** adding "
                f"**{secondary_driver['impact_hours']:+.1f} hrs**."
            )
            
            # Contextual Logistics Observations
            if pred_res["berth_window_miss_risk"]:
                narrative_lines.append(
                    "• **Berth Window Risk**: Vessel arrival schedule exceeds terminal tolerance, triggering "
                    "a missed berthing window penalty and non-linear outer anchorage dwell."
                )
            if "Red" in customs_tier:
                narrative_lines.append(
                    "• **Customs Alert**: Consignment flagged for **Red Channel Physical Examination**, "
                    "adding heavy inspection dwell at destination terminal."
                )
            if weather_sev > 0.5:
                narrative_lines.append(
                    f"• **Weather Impact**: High sea state / storm index ({weather_sev:.2f}) required vessel speed throttling."
                )
            if c_buffer < -0.5:
                narrative_lines.append(
                    f"• **Buffer Absorption**: Ocean transit absorbed {abs(c_buffer):.1f} hours through dynamic vessel speed adjustment."
                )
        else:
            narrative_lines.append(
                f"**Status: On Schedule.** Estimated arrival {pred_res['predicted_eta']}, "
                f"Net variance: {pred_delay:+.1f} hours. Low delay risk ({pred_res['delay_probability']}%)."
            )
            narrative_lines.append(
                f"• Buffer margins across {origin_port} to {dest_port} are successfully absorbing minor fluctuations."
            )

        # Actionable Operational Recommendations
        recommendations = []
        if primary_driver["factor"] == "Port Congestion & Berth Queue" or pred_res["berth_window_miss_risk"]:
            recommendations.append("Priority Berthing Request: Notify port agent 48h prior to reserve alternate off-peak berthing window.")
            recommendations.append("Virtual Queuing: Advise vessel to optimize cruising speed to meet designated slot without anchorage idling.")
        elif primary_driver["factor"] == "Customs Inspection Dwell":
            recommendations.append("Pre-Clearance Protocol: Submit electronic manifest and customs pre-declaration prior to vessel discharge.")
            recommendations.append("B/L Expedite: Coordinate bonded warehouse transfer to avoid demurrage at marine terminal.")
        elif primary_driver["factor"] == "Origin Departure Lag":
            recommendations.append("Feeder Buffer Expansion: Extend origin gate cut-off buffer by 6 hours for future bookings on this lane.")
            recommendations.append("Carrier SLA Review: Escalate repeated late feeder handovers with carrier operations desk.")
        elif primary_driver["factor"] == "Inland Transport & Traffic":
            recommendations.append("Off-Peak Drayage: Dispatch inland container drayage during evening/night non-curfew window.")
            recommendations.append("Intermodal Rail Alternative: Switch final leg to dedicated freight rail shuttle if road congestion persists.")
        else:
            recommendations.append("Standard Monitoring: Maintain standard automated IoT/AIS milestone tracking.")

        return {
            "prediction": pred_res,
            "contributions": contributions,
            "primary_driver": primary_driver,
            "secondary_driver": secondary_driver,
            "narrative": "\n".join(narrative_lines),
            "recommendations": recommendations,
        }
