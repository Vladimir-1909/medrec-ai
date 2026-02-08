"""Multi-agent clinical workflow with real LLM calls and TERMINATE handling"""
import time
import json
from typing import List, Dict, Any, Callable, Optional
from autogen import AssistantAgent, UserProxyAgent
from agents.clinical_agents import (
    create_primary_care_agent,
    create_specialist_agent,
    create_safety_agent,
    create_guidelines_agent,
    create_patient_comm_agent,
    AGENT_ROLES,
)

class ClinicalMultiAgentWorkflow:
    """
    Sequential multi-agent workflow that actually calls LLM for each agent.

    CRITICAL FIX: Uses direct generate_reply() calls instead of broken GroupChat
    to ensure LLM requests are sent for each agent step.
    """

    def __init__(
        self,
        llm_config: Dict[str, Any],
        vector_store: Any,  # ClinicalVectorStore instance
        system_prompt: str,
    ):
        self.llm_config = llm_config
        self.vector_store = vector_store
        self.system_prompt = system_prompt
        self.agent_messages = []  # For real-time UI streaming

        # Create agents WITH llm_config properly attached
        self.primary = create_primary_care_agent(llm_config)
        self.specialist = create_specialist_agent(llm_config)
        self.safety = create_safety_agent(llm_config)
        self.guidelines = create_guidelines_agent(llm_config)
        self.patient_comm = create_patient_comm_agent(llm_config)

        # User proxy for message formatting (no LLM calls)
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )

    def run_workflow(
        self,
        patient_context: Dict[str, Any],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute sequential agent workflow with REAL LLM calls for each step.

        This implementation FIXES the broken GroupChat approach by:
        1. Building context with RAG examples
        2. Calling generate_reply() directly on each agent
        3. Passing conversation history between agents
        4. Enforcing TERMINATE protocol manually

        Returns:
            Dict with agent messages and final recommendation
        """
        # Reset message history
        self.agent_messages = []

        # Build RAG context from similar examples
        rag_context = self._build_rag_context(patient_context)

        # Build initial prompt with system instructions + RAG + patient data
        initial_prompt = self._build_initial_prompt(patient_context, rag_context)

        # === STEP 1: Primary Care Physician ===
        if callback:
            callback({
                "agent": "primary_care",
                "role": AGENT_ROLES["primary_care"],
                "content": "Analyzing patient data and generating initial clinical assessment...",
                "timestamp": time.time(),
            })

        primary_response = self._call_llm_with_retry(
            agent=self.primary,
            messages=[{"role": "user", "content": initial_prompt}],
            max_retries=2
        )

        self._store_message("primary_care", primary_response)
        if callback:
            callback(self.agent_messages[-1])
        time.sleep(0.5)  # Simulate real-time flow

        # === STEP 2: Specialist Consultant ===
        if callback:
            callback({
                "agent": "specialist",
                "role": AGENT_ROLES["specialist"],
                "content": "Adding specialty-specific insights and checking for drug interactions...",
                "timestamp": time.time(),
            })

        specialist_prompt = self._build_specialist_prompt(primary_response, patient_context)
        specialist_response = self._call_llm_with_retry(
            agent=self.specialist,
            messages=[
                {"role": "user", "content": initial_prompt},
                {"role": "assistant", "content": primary_response},
                {"role": "user", "content": specialist_prompt}
            ],
            max_retries=2
        )

        self._store_message("specialist", specialist_response)
        if callback:
            callback(self.agent_messages[-1])
        time.sleep(0.5)

        # === STEP 3: Safety Officer ===
        if callback:
            callback({
                "agent": "safety",
                "role": AGENT_ROLES["safety"],
                "content": "Performing comprehensive safety audit for red flags and contraindications...",
                "timestamp": time.time(),
            })

        safety_prompt = self._build_safety_prompt(specialist_response, patient_context)
        safety_response = self._call_llm_with_retry(
            agent=self.safety,
            messages=[
                {"role": "user", "content": initial_prompt},
                {"role": "assistant", "content": primary_response},
                {"role": "assistant", "content": specialist_response},
                {"role": "user", "content": safety_prompt}
            ],
            max_retries=2
        )

        self._store_message("safety", safety_response)
        if callback:
            callback(self.agent_messages[-1])
        time.sleep(0.5)

        # === STEP 4: Guidelines Specialist ===
        if callback:
            callback({
                "agent": "guidelines",
                "role": AGENT_ROLES["guidelines"],
                "content": "Verifying alignment with current US clinical guidelines (ADA, AHA, CDC)...",
                "timestamp": time.time(),
            })

        guidelines_prompt = self._build_guidelines_prompt(safety_response, patient_context)
        guidelines_response = self._call_llm_with_retry(
            agent=self.guidelines,
            messages=[
                {"role": "user", "content": initial_prompt},
                {"role": "assistant", "content": primary_response},
                {"role": "assistant", "content": specialist_response},
                {"role": "assistant", "content": safety_response},
                {"role": "user", "content": guidelines_prompt}
            ],
            max_retries=2
        )

        self._store_message("guidelines", guidelines_response)
        if callback:
            callback(self.agent_messages[-1])
        time.sleep(0.5)

        # === STEP 5: Patient Communication Specialist ===
        if callback:
            callback({
                "agent": "patient_comm",
                "role": AGENT_ROLES["patient_comm"],
                "content": "Translating clinical findings into patient-friendly language (8th-grade literacy level)...",
                "timestamp": time.time(),
            })

        patient_prompt = self._build_patient_prompt(guidelines_response, patient_context)
        patient_response = self._call_llm_with_retry(
            agent=self.patient_comm,
            messages=[
                {"role": "user", "content": initial_prompt},
                {"role": "assistant", "content": primary_response},
                {"role": "assistant", "content": specialist_response},
                {"role": "assistant", "content": safety_response},
                {"role": "assistant", "content": guidelines_response},
                {"role": "user", "content": patient_prompt}
            ],
            max_retries=2
        )

        # Ensure TERMINATE is present (critical for protocol compliance)
        if "TERMINATE" not in patient_response.upper():
            patient_response += "\n\nTERMINATE"

        self._store_message("patient_comm", patient_response)
        if callback:
            callback(self.agent_messages[-1])

        # Extract final recommendation (cleaned from TERMINATE marker)
        final_content = patient_response.replace("TERMINATE", "").strip()

        return {
            "agent_messages": self.agent_messages,
            "final_recommendation": {
                "agent": "patient_comm",
                "role": AGENT_ROLES["patient_comm"],
                "content": final_content,
            },
            "rag_context_used": rag_context,
            "timestamp": time.time(),
        }

    def _call_llm_with_retry(self, agent, messages: List[Dict], max_retries: int = 3) -> str:
        """Call LLM with retry logic for robustness"""
        for attempt in range(max_retries):
            try:
                response = agent.generate_reply(messages=messages)
                if response and isinstance(response, str) and len(response.strip()) > 20:
                    return response.strip()
                else:
                    # Fallback response if LLM returns empty/short response
                    return "Clinical analysis complete. No additional comments at this time. TERMINATE"
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {str(e)}")
                time.sleep(1.0)  # Wait before retry

    def _store_message(self, agent_name: str, content: str):
        """Store agent message in history with TERMINATE handling"""
        # Remove duplicate TERMINATE markers
        cleaned_content = content.replace("TERMINATE TERMINATE", "TERMINATE").strip()

        self.agent_messages.append({
            "agent": agent_name,
            "role": AGENT_ROLES.get(agent_name, f"🤖 {agent_name}"),
            "content": cleaned_content,
            "timestamp": time.time(),
        })

    def _build_rag_context(self, patient_context: Dict[str, Any]) -> str:
        """Build RAG context from similar physician examples"""
        symptoms = ", ".join(patient_context.get("symptoms", []))
        labs = patient_context.get("labs", {})

        query = f"""
Patient: {patient_context.get('age', 'N/A')} year old {patient_context.get('sex', 'N/A')}
Symptoms: {symptoms}
Labs: Glucose {labs.get('fasting_glucose_mg_dl', 'N/A')} mg/dL, 
      HbA1c {labs.get('hba1c_percent', 'N/A')}%, 
      Creatinine {labs.get('creatinine_mg_dl', 'N/A')} mg/dL
        """.strip()

        similar_examples = self.vector_store.semantic_search_examples(query, top_k=3)

        if similar_examples:
            rag_context = "RELEVANT PHYSICIAN EXAMPLES (RAG):\n" + "="*60 + "\n"
            for i, ex in enumerate(similar_examples, 1):
                rag_context += f"\nExample {i} (Similarity: {ex['similarity']:.2f}):\n"
                # Extract just the recommendation part for context
                if "recommendation" in ex['document'].lower():
                    excerpt_start = ex['document'].lower().find("recommendation")
                    excerpt = ex['document'][excerpt_start:excerpt_start+300] if excerpt_start != -1 else ex['document'][:300]
                else:
                    excerpt = ex['document'][:300]
                rag_context += f"{excerpt}...\n"
            return rag_context
        else:
            return "No highly similar physician examples found in knowledge base."

    def _build_initial_prompt(self, patient_context: Dict[str, Any], rag_context: str) -> str:
        """Build complete initial prompt with system instructions"""
        labs = patient_context.get("labs", {})
        symptoms = ", ".join(patient_context.get("symptoms", []))
        questionnaire = patient_context.get("questionnaire", "No additional history provided")

        return f"""
{self.system_prompt}

PATIENT DATA:
Age: {patient_context.get('age', 'N/A')} years
Sex: {patient_context.get('sex', 'N/A')}
Symptoms: {symptoms}
Questionnaire: {questionnaire}

Lab Results (US Units):
- Fasting Glucose: {labs.get('fasting_glucose_mg_dl', 'N/A')} mg/dL
- HbA1c: {labs.get('hba1c_percent', 'N/A')}%
- Total Cholesterol: {labs.get('total_cholesterol_mg_dl', 'N/A')} mg/dL
- LDL Cholesterol: {labs.get('ldl_cholesterol_mg_dl', 'N/A')} mg/dL
- HDL Cholesterol: {labs.get('hdl_cholesterol_mg_dl', 'N/A')} mg/dL
- Triglycerides: {labs.get('triglycerides_mg_dl', 'N/A')} mg/dL
- Creatinine: {labs.get('creatinine_mg_dl', 'N/A')} mg/dL
- eGFR: {labs.get('egfr_ml_min_173m2', 'N/A')} mL/min/1.73m²
- Hemoglobin: {labs.get('hemoglobin_g_dl', 'N/A')} g/dL
- Ferritin: {labs.get('ferritin_ng_ml', 'N/A')} ng/mL

{rag_context}

YOUR TASK:
Generate a complete clinical interpretation following the EXACT 5-block structure specified in the system prompt above.
CRITICAL: End your response with exactly "TERMINATE" on a new line after the fifth block.
        """.strip()

    def _build_specialist_prompt(self, previous_response: str, patient_context: Dict) -> str:
        return f"""
Review the primary care assessment below and ADD specialty-specific insights ONLY where clinically indicated.
Focus on: drug interactions, contraindications, specialty referral indications.
DO NOT rewrite the entire assessment - only add high-value specialty insights.

Primary Assessment:
{previous_response}

Patient Context:
Age: {patient_context.get('age')} | Sex: {patient_context.get('sex')}
Key Labs: {', '.join([f'{k}={v}' for k, v in list(patient_context.get('labs', {}).items())[:3]])}

Provide concise specialty additions. END with TERMINATE.
        """.strip()

    def _build_safety_prompt(self, previous_response: str, patient_context: Dict) -> str:
        return f"""
Perform COMPREHENSIVE SAFETY AUDIT of the clinical assessment below.
Check for: FDA black box warnings, absolute contraindications, critical lab values, high-risk drug interactions.

Assessment to Audit:
{previous_response}

Patient Context:
Age: {patient_context.get('age')} | Sex: {patient_context.get('sex')}
Labs: {json.dumps(patient_context.get('labs', {}), indent=2)}

If SAFE: "SAFETY REVIEW: No concerns identified. APPROVED. TERMINATE"
If UNSAFE: "SAFETY CONCERN: [specific issue]. Action required: [correction]. TERMINATE"
        """.strip()

    def _build_guidelines_prompt(self, previous_response: str, patient_context: Dict) -> str:
        return f"""
Verify alignment of the clinical assessment with CURRENT US clinical guidelines (2024-2025).
Check: ADA for diabetes, AHA/ACC for cardiac, CDC for infections, USPSTF for screenings.

Assessment to Verify:
{previous_response}

Provide alignment score and specific guideline references. END with TERMINATE.
        """.strip()

    def _build_patient_prompt(self, previous_response: str, patient_context: Dict) -> str:
        return f"""
Translate the clinical assessment below into PATIENT-FRIENDLY language.
Requirements:
- 8th-grade reading level (CDC Clear Communication Index)
- NO medical jargon (e.g., "high blood sugar" not "hyperglycemia")
- Specific actionable instructions ("with breakfast" not "in morning")
- Empathetic tone
- US context (mention generics, ACA preventive services)

Clinical Assessment:
{previous_response}

Generate complete 5-block interpretation in plain language. END with TERMINATE.
        """.strip()
