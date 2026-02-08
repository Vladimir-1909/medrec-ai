from __future__ import annotations

import io
import pdfplumber
import PyPDF2

from typing import Dict


class DocumentLoader:
    """Loader for clinical documents (PDF lab reports, TXT notes)"""

    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> str:
        """
        Extract text from PDF file (lab reports, clinical notes).

        Uses pdfplumber first (better table extraction), falls back to PyPDF2.
        """
        try:
            # Try pdfplumber first (better for tables/structured data)
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                if text.strip():
                    return text.strip()
        except Exception:
            pass

        try:
            # Fallback to PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
            return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    @staticmethod
    def parse_lab_results(text: str) -> Dict[str, Dict]:
        """
        Parse common lab values from extracted text using pattern matching.

        Returns structured lab results dictionary.
        """
        labs = {}
        text_lower = text.lower()

        # Glucose (mg/dL)
        if "glucose" in text_lower or "blood sugar" in text_lower:
            # Simple pattern matching - in production use regex/NLP
            import re
            glucose_match = re.search(r'glucose\D*(\d+\.?\d*)', text_lower)
            if glucose_match:
                labs["glucose_mg_dl"] = float(glucose_match.group(1))

        # HbA1c (%)
        if "a1c" in text_lower or "hba1c" in text_lower:
            hba1c_match = re.search(r'(a1c|hba1c)\D*(\d+\.?\d*)', text_lower)
            if hba1c_match:
                labs["hba1c_percent"] = float(hba1c_match.group(2))

        # Creatinine (mg/dL)
        if "creatinine" in text_lower:
            creatinine_match = re.search(r'creatinine\D*(\d+\.?\d*)', text_lower)
            if creatinine_match:
                labs["creatinine_mg_dl"] = float(creatinine_match.group(1))

        # Cholesterol (mg/dL)
        if "cholesterol" in text_lower:
            cholesterol_match = re.search(r'cholesterol\D*(\d+\.?\d*)', text_lower)
            if cholesterol_match:
                labs["cholesterol_mg_dl"] = float(cholesterol_match.group(1))

        return {"labs": labs, "raw_text": text[:1000] + "..."}
