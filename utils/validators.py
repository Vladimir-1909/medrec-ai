"""Validation utilities for clinical laboratory results"""
from __future__ import annotations

from typing import Dict, List


def validate_lab_results(labs: Dict[str, float]) -> List[str]:
    """
    Validate lab results against US critical value thresholds.

    Returns list of safety flags for critical values.
    """
    flags = []

    # Glucose (mg/dL)
    glucose = labs.get("glucose_mg_dl")
    if glucose is not None:
        if glucose > 250:
            flags.append("🚨 CRITICAL: Glucose >250 mg/dL (DKA risk)")
        elif glucose > 180:
            flags.append("⚠️ HIGH: Glucose >180 mg/dL (poor control)")
        elif glucose < 70:
            flags.append("🚨 CRITICAL: Glucose <70 mg/dL (hypoglycemia)")

    # HbA1c (%)
    hba1c = labs.get("hba1c_percent")
    if hba1c is not None:
        if hba1c > 9.0:
            flags.append("⚠️ HIGH: HbA1c >9% (poor diabetes control)")

    # Creatinine (mg/dL) - renal function proxy
    creatinine = labs.get("creatinine_mg_dl")
    if creatinine is not None:
        if creatinine > 1.5:
            flags.append("⚠️ HIGH: Creatinine >1.5 mg/dL (possible renal impairment)")
        elif creatinine > 2.0:
            flags.append("🚨 CRITICAL: Creatinine >2.0 mg/dL (significant renal impairment)")

    return flags
