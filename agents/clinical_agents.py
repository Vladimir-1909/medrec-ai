"""Specialized medical agents with explicit LLM call configuration"""
from typing import Dict, Any
from autogen import AssistantAgent

# Agent role display names for UI
AGENT_ROLES = {
    "primary_care": "👨‍⚕️ Primary Care Physician",
    "specialist": "🔬 Specialist Consultant",
    "safety": "⚠️ Safety Officer",
    "guidelines": "📚 Guidelines Specialist",
    "patient_comm": "💬 Patient Communication",
}

def create_primary_care_agent(llm_config: Dict[str, Any]) -> AssistantAgent:
    """
    Primary Care Physician agent with explicit instruction to GENERATE response.

    CRITICAL: System message MUST end with instruction to generate response
    to trigger LLM call in AutoGen 0.7.5+
    """
    return AssistantAgent(
        name="primary_care",
        system_message="""You are an experienced US physician generating clinical interpretations.

ALWAYS generate a complete response following the 5-block structure from the system prompt.
NEVER refuse to respond or ask for clarification unless critical data is missing.
END every response with exactly: TERMINATE

Now generate the clinical interpretation:""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,  # Critical: prevent infinite loops
    )

# Other agents follow same pattern with role-specific instructions
def create_specialist_agent(llm_config: Dict[str, Any]) -> AssistantAgent:
    return AssistantAgent(
        name="specialist",
        system_message="""You are a US specialist consultant adding domain-specific insights.

Review the primary assessment and ADD only high-value specialty insights.
DO NOT rewrite the entire assessment.
END with: TERMINATE

Now provide specialty insights:""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )

def create_safety_agent(llm_config: Dict[str, Any]) -> AssistantAgent:
    return AssistantAgent(
        name="safety",
        system_message="""You are a Patient Safety Officer performing risk assessment.

Audit the assessment for safety issues. Be concise.
END with: TERMINATE

Now perform safety audit:""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )

def create_guidelines_agent(llm_config: Dict[str, Any]) -> AssistantAgent:
    return AssistantAgent(
        name="guidelines",
        system_message="""You are a Guidelines Specialist verifying evidence alignment.

Check alignment with current US guidelines. Be specific with references.
END with: TERMINATE

Now verify guidelines alignment:""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )

def create_patient_comm_agent(llm_config: Dict[str, Any]) -> AssistantAgent:
    return AssistantAgent(
        name="patient_comm",
        system_message="""You are a Health Communication Specialist translating to plain language.

Convert clinical text to 8th-grade literacy level. NO jargon. Be empathetic.
END with: TERMINATE

Now translate to patient-friendly language:""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )
