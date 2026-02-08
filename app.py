"""Streamlit UI for MedRec AI - Multi-agent clinical decision support with RAG"""
from __future__ import annotations

import streamlit as st
import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Callable

# CRITICAL: Clean proxy environment variables BEFORE importing autogen
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(key, None)

from llm.local_llm import get_llm_config
from rag.vector_store import ClinicalVectorStore
from rag.document_loader import DocumentLoader
from agents.workflow import ClinicalMultiAgentWorkflow
from utils.validators import validate_lab_results

# Set page configuration
st.set_page_config(
    page_title="🏥 MedRec AI: Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title and caption
st.title("🏥 MedRec AI: Multi-Agent Clinical Decision Support")
st.caption("100% local | RAG-powered | Physician-guided recommendations")

# === SESSION STATE INITIALIZATION ===
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = None
if "examples_loaded" not in st.session_state:
    st.session_state.examples_loaded = False
if "knowledge_loaded" not in st.session_state:
    st.session_state.knowledge_loaded = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "final_recommendation" not in st.session_state:
    st.session_state.final_recommendation = None
if "consultation_history" not in st.session_state:
    st.session_state.consultation_history = []


if st.button("🔍 Test LLM Connection"):
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"
        )
        resp = client.chat.completions.create(
            model="medgemma-1.5-4b-it",
            messages=[{"role": "user", "content": "Hello, are you working?"}],
            max_tokens=50
        )
        st.success(f"✅ LLM responded: {resp.choices[0].message.content}")
    except Exception as e:
        st.error(f"❌ LLM connection failed: {str(e)}")

# === LM STUDIO CONNECTION CHECK ===
def check_lmstudio() -> bool:
    """
    Check if LM Studio is running and MedGemma model is available.

    Returns:
        bool: True if LM Studio is running with MedGemma loaded
    """
    try:
        resp = requests.get("http://localhost:1234/v1/models", timeout=3)
        if resp.status_code != 200:
            return False

        # Check if MedGemma model is loaded
        models = resp.json().get("data", [])
        for model in models:
            if "medgemma" in model.get("id", "").lower():
                return True
        return False
    except Exception:
        return False


lmstudio_ok = check_lmstudio()

# === SIDEBAR: SETUP WIZARD ===
with st.sidebar:
    st.header("⚙️ Setup Wizard")

    # Step 1: System Prompt
    st.subheader("Step 1: System Prompt")
    st.info("Define how AI should structure clinical interpretations (5-block format)")

    prompt_file = st.file_uploader(
        "Upload system prompt (TXT)",
        type=["txt"],
        key="prompt_uploader",
        help="Text file defining the 5-block interpretation structure and clinical standards"
    )

    if prompt_file:
        st.session_state.system_prompt = prompt_file.read().decode("utf-8")
        st.success("✅ System prompt loaded")
        st.caption(f"Length: {len(st.session_state.system_prompt)} characters")

    # Step 2: Physician Examples
    st.subheader("Step 2: Physician Examples")
    st.info("Upload 20+ examples of your clinical interpretations for RAG")

    examples_file = st.file_uploader(
        "Upload examples (JSON)",
        type=["json"],
        key="examples_uploader",
        help="JSON array of {patient_context, physician_recommendation} pairs"
    )

    if examples_file:
        try:
            examples = json.load(examples_file)
            if isinstance(examples, list):
                if len(examples) < 20:
                    st.warning(f"⚠️ Only {len(examples)} examples provided. Minimum 20 recommended for effective RAG.")
                else:
                    st.success(f"✅ {len(examples)} examples loaded")

                # Initialize vector store if not exists
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = ClinicalVectorStore()

                # Save examples to temp file
                temp_path = Path("./data/temp_examples.json")
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(examples, f, ensure_ascii=False, indent=2)

                # Ingest into vector store
                count = st.session_state.vector_store.ingest_physician_examples(str(temp_path))
                st.session_state.examples_loaded = True
                st.success(f"✅ {count} examples ingested into RAG system")
            else:
                st.error("❌ Invalid JSON format. Expected array of examples.")
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON syntax: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

    # Step 3: Knowledge Base (Optional)
    st.subheader("Step 3: Knowledge Base (Optional)")
    st.info("Upload clinical guidelines (PDF/TXT) for enhanced RAG")

    kb_files = st.file_uploader(
        "Upload guidelines (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="kb_uploader"
    )

    if kb_files and st.session_state.vector_store:
        with st.spinner("Processing knowledge base documents..."):
            documents = []
            doc_names = []

            for file in kb_files:
                try:
                    if file.type == "application/pdf":
                        text = DocumentLoader.extract_text_from_pdf(file.read())
                    else:
                        text = file.read().decode("utf-8")

                    documents.append(text)
                    doc_names.append(file.name)
                except Exception as e:
                    st.warning(f"⚠️ Failed to process {file.name}: {str(e)}")

            if documents:
                count = st.session_state.vector_store.ingest_knowledge_documents(documents, doc_names)
                st.session_state.knowledge_loaded = True
                st.success(f"✅ {count} knowledge documents ingested")
            else:
                st.warning("⚠️ No documents were successfully processed")

    # Step 4: Initialize Agents
    st.subheader("Step 4: Initialize Agents")

    # Display status indicators
    if not lmstudio_ok:
        st.error("❌ LM Studio not running on port 1234")
        st.info("""
        **Setup LM Studio:**
        1. Download: https://lmstudio.ai/
        2. Load model: `medgemma-1.5-4b-it`
        3. Click: Server → Start Server (default port 1234)
        """)
    elif not st.session_state.system_prompt:
        st.warning("⚠️ Upload system prompt first")
    elif not st.session_state.examples_loaded:
        st.warning("⚠️ Upload physician examples first (minimum 20 recommended)")
    else:
        if st.button("🚀 Initialize Multi-Agent System", type="primary", use_container_width=True):
            with st.spinner("Initializing 5 specialized medical agents..."):
                try:
                    # Get LLM config
                    llm_config = get_llm_config()

                    # Create workflow
                    st.session_state.workflow = ClinicalMultiAgentWorkflow(
                        llm_config=llm_config,
                        vector_store=st.session_state.vector_store,
                        system_prompt=st.session_state.system_prompt
                    )

                    st.success("✅ Multi-agent system initialized successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
                    st.exception(e)

    # System status
    st.markdown("---")
    st.subheader("📊 System Status")
    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.metric("LM Studio", "✅ Running" if lmstudio_ok else "❌ Not Running")
        st.metric("System Prompt", "✅ Loaded" if st.session_state.system_prompt else "⚠️ Missing")

    with status_col2:
        st.metric("Examples", f"✅ {st.session_state.vector_store.examples_collection.count()}" if st.session_state.examples_loaded else "⚠️ Missing")
        st.metric("Knowledge Base", f"✅ {st.session_state.vector_store.knowledge_collection.count()}" if st.session_state.knowledge_loaded else "⚪ Optional")

# === MAIN INTERFACE: PATIENT CONSULTATION ===
if st.session_state.workflow:
    tab1, tab2, tab3 = st.tabs(["💬 New Consultation", "📁 Consultation History", "ℹ️ How It Works"])

    with tab1:
        st.subheader("📋 Patient Data Input")

        # Input method selection
        input_method = st.radio(
            "How to provide patient data:",
            ["📝 Manual entry", "📄 Upload lab report (PDF)"],
            horizontal=True,
            help="Choose manual entry for structured data or PDF upload for lab reports"
        )

        patient_context: Dict[str, Any] = {}

        if input_method == "📝 Manual entry":
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Patient Demographics**")
                patient_context["age"] = st.number_input("Age (years)", min_value=0, max_value=120, value=52, step=1)
                patient_context["sex"] = st.selectbox("Sex", ["Male", "Female", "Other/Prefer not to say"])

                st.write("**Symptoms & Complaints**")
                symptoms_input = st.text_area(
                    "Symptoms (comma separated)",
                    "fatigue, increased thirst, blurred vision",
                    height=100,
                    help="Enter all reported symptoms separated by commas"
                )
                patient_context["symptoms"] = [s.strip() for s in symptoms_input.split(",") if s.strip()]

                st.write("**Medical History**")
                patient_context["questionnaire"] = st.text_area(
                    "Additional history (optional)",
                    "No known chronic conditions. Non-smoker. Occasional alcohol use.",
                    height=80,
                    help="Relevant medical history, medications, family history, lifestyle factors"
                )

            with col2:
                st.write("**Laboratory Results (US Units)**")

                # Glucose panel
                st.subheader("Glucose Metabolism")
                patient_context["labs"] = {}
                patient_context["labs"]["fasting_glucose_mg_dl"] = st.number_input(
                    "Fasting Glucose (mg/dL)",
                    min_value=0.0, max_value=600.0, value=92.0, step=1.0,
                    help="Normal: 70-99 mg/dL | Prediabetes: 100-125 mg/dL | Diabetes: ≥126 mg/dL"
                )
                patient_context["labs"]["hba1c_percent"] = st.number_input(
                    "HbA1c (%)",
                    min_value=0.0, max_value=15.0, value=5.5, step=0.1,
                    help="Normal: <5.7% | Prediabetes: 5.7-6.4% | Diabetes: ≥6.5%"
                )

                # Lipid panel
                st.subheader("Lipid Profile")
                patient_context["labs"]["total_cholesterol_mg_dl"] = st.number_input(
                    "Total Cholesterol (mg/dL)",
                    min_value=0.0, max_value=500.0, value=190.0, step=1.0
                )
                patient_context["labs"]["ldl_cholesterol_mg_dl"] = st.number_input(
                    "LDL Cholesterol (mg/dL)",
                    min_value=0.0, max_value=300.0, value=110.0, step=1.0,
                    help="Optimal: <100 mg/dL | Near optimal: 100-129 mg/dL"
                )
                patient_context["labs"]["hdl_cholesterol_mg_dl"] = st.number_input(
                    "HDL Cholesterol (mg/dL)",
                    min_value=0.0, max_value=150.0, value=50.0, step=1.0,
                    help="Low (risk factor): <40 mg/dL men, <50 mg/dL women"
                )
                patient_context["labs"]["triglycerides_mg_dl"] = st.number_input(
                    "Triglycerides (mg/dL)",
                    min_value=0.0, max_value=1000.0, value=150.0, step=1.0,
                    help="Normal: <150 mg/dL | Borderline: 150-199 mg/dL"
                )

                # Renal function
                st.subheader("Renal Function")
                patient_context["labs"]["creatinine_mg_dl"] = st.number_input(
                    "Creatinine (mg/dL)",
                    min_value=0.0, max_value=15.0, value=0.9, step=0.1
                )
                patient_context["labs"]["egfr_ml_min_173m2"] = st.number_input(
                    "eGFR (mL/min/1.73m²)",
                    min_value=0.0, max_value=200.0, value=90.0, step=1.0,
                    help="Normal: ≥90 | Mild reduction: 60-89 | Moderate: 30-59"
                )

                # Hematology
                st.subheader("Hematology")
                patient_context["labs"]["hemoglobin_g_dl"] = st.number_input(
                    "Hemoglobin (g/dL)",
                    min_value=0.0, max_value=25.0, value=14.0, step=0.1,
                    help="Men: 13.8-17.2 | Women: 12.1-15.1"
                )
                patient_context["labs"]["ferritin_ng_ml"] = st.number_input(
                    "Ferritin (ng/mL)",
                    min_value=0.0, max_value=1000.0, value=80.0, step=1.0,
                    help="Iron deficiency: <30 ng/mL | Normal: 30-400"
                )

        else:  # PDF upload
            st.info("📄 Upload a PDF lab report. The system will extract laboratory values automatically.")

            uploaded_file = st.file_uploader(
                "Upload lab report PDF",
                type=["pdf"],
                help="PDF file containing laboratory test results"
            )

            if uploaded_file:
                with st.spinner("Extracting data from PDF..."):
                    try:
                        text = DocumentLoader.extract_text_from_pdf(uploaded_file.read())
                        parsed = DocumentLoader.parse_lab_results(text)

                        st.success("✅ Lab report processed successfully!")

                        # Show extracted text preview
                        with st.expander("📄 Extracted text preview", expanded=False):
                            st.text_area("Full extracted text", parsed.get("raw_text", ""), height=200)

                        # Show parsed labs
                        st.write("**Parsed laboratory values:**")
                        if parsed.get("labs"):
                            for key, value in parsed["labs"].items():
                                st.write(f"- {key.replace('_', ' ').title()}: {value}")
                        else:
                            st.warning("⚠️ No laboratory values were automatically detected. Please enter manually.")

                        # Create patient context with defaults
                        patient_context = {
                            "age": st.number_input("Patient age (from report)", 0, 120, 52, key="pdf_age"),
                            "sex": st.selectbox("Patient sex", ["Male", "Female", "Other"], key="pdf_sex"),
                            "symptoms": [],
                            "questionnaire": "Lab report uploaded via PDF",
                            "labs": parsed.get("labs", {})
                        }

                        # Allow manual override of parsed values
                        if st.checkbox("✏️ Edit parsed values manually"):
                            st.write("**Edit laboratory values:**")
                            for key in list(patient_context["labs"].keys()):
                                new_value = st.number_input(
                                    f"{key.replace('_', ' ').title()}",
                                    value=float(patient_context["labs"][key]),
                                    key=f"edit_{key}"
                                )
                                patient_context["labs"][key] = new_value

                    except Exception as e:
                        st.error(f"❌ PDF processing failed: {str(e)}")
                        st.info("Please enter laboratory values manually above.")

        # Safety validation
        if patient_context.get("labs"):
            safety_flags = validate_lab_results(patient_context["labs"])
            if safety_flags:
                st.warning("⚠️ **Safety flags detected:**")
                for flag in safety_flags:
                    st.write(f"- {flag}")
                st.info("💡 These flags will be highlighted in the clinical interpretation.")

        # Generate recommendation button
        st.markdown("---")
        if st.button("🧠 Generate Multi-Agent Clinical Interpretation", type="primary", disabled=not patient_context, use_container_width=True):
            if not patient_context.get("symptoms") and not patient_context.get("labs"):
                st.error("❌ Please provide at least symptoms or laboratory results")
            else:
                # Reset previous messages
                st.session_state.agent_messages = []
                st.session_state.final_recommendation = None

                # Display patient summary
                with st.expander("📋 Patient Summary", expanded=True):
                    st.write(f"**Age:** {patient_context.get('age')} years")
                    st.write(f"**Sex:** {patient_context.get('sex')}")
                    if patient_context.get("symptoms"):
                        st.write(f"**Symptoms:** {', '.join(patient_context['symptoms'])}")
                    if patient_context.get("questionnaire"):
                        st.write(f"**History:** {patient_context['questionnaire'][:100]}...")

                    st.write("**Laboratory Values:**")
                    for key, value in patient_context.get("labs", {}).items():
                        st.write(f"- {key.replace('_', ' ').title()}: {value}")

                # Real-time message display container
                st.subheader("🔄 Multi-Agent Collaboration in Real-Time")
                messages_container = st.container()

                # Callback for real-time streaming
                def stream_message(msg: dict):
                    """Callback function to display agent messages in real-time"""
                    with messages_container:
                        with st.chat_message("assistant"):
                            # Display agent role with emoji
                            st.markdown(f"**{msg['role']}**")

                            # Clean content: remove TERMINATE marker and format
                            display_content = msg["content"].replace("TERMINATE", "").strip()

                            if display_content:
                                # Format the 5-block structure nicely
                                if any(block in display_content.lower() for block in ["systems and organs", "detected abnormalities", "additional comments", "recommendations for additional tests", "recommendations for specialist"]):
                                    # This is the final recommendation - format as structured text
                                    st.markdown(display_content)
                                else:
                                    st.markdown(display_content)
                            else:
                                st.markdown("*Processing...*")

                # Run workflow with real-time streaming
                with st.spinner("🔄 Agents collaborating... (Primary Care → Specialist → Safety → Guidelines → Patient Communication)"):
                    try:
                        result = st.session_state.workflow.run_workflow(
                            patient_context=patient_context,
                            callback=stream_message
                        )

                        st.session_state.agent_messages = result["agent_messages"]
                        st.session_state.final_recommendation = result["final_recommendation"]

                        # Show completion message
                        st.success("✅ Multi-agent clinical interpretation complete!")

                        # Display final recommendation prominently
                        st.subheader("🎯 Final Clinical Interpretation")
                        with st.expander("📄 View Complete Interpretation", expanded=True):
                            st.markdown(f"**Generated by:** {st.session_state.final_recommendation['role']}")
                            st.markdown("---")
                            st.markdown(st.session_state.final_recommendation["content"])

                            # Add download button for the interpretation
                            interpretation_text = st.session_state.final_recommendation["content"]
                            st.download_button(
                                label="💾 Download Interpretation (TXT)",
                                data=interpretation_text,
                                file_name=f"clinical_interpretation_{int(time.time())}.txt",
                                mime="text/plain"
                            )

                        # Save to consultation history
                        st.session_state.consultation_history.append({
                            "timestamp": time.time(),
                            "patient": patient_context,
                            "recommendation": st.session_state.final_recommendation["content"],
                            "agents_involved": len(st.session_state.agent_messages),
                            "rag_matches": result.get("rag_context_used", "N/A")
                        })

                        # Show summary statistics
                        st.info(f"💡 **Analysis Summary:** {len(st.session_state.agent_messages)} agent messages processed | RAG examples utilized: {len(result.get('rag_context_used', []))}")

                    except Exception as e:
                        st.error(f"❌ Workflow execution failed: {str(e)}")
                        st.exception(e)

    with tab2:
        st.subheader("📁 Consultation History")

        if not st.session_state.consultation_history:
            st.info("📭 No consultations yet. Complete a consultation to see history here.")
            st.info("💡 Tip: After generating an interpretation, it will be saved automatically to this history.")
        else:
            # Statistics
            total_consults = len(st.session_state.consultation_history)
            avg_ages = [c["patient"].get("age", 0) for c in st.session_state.consultation_history if c["patient"].get("age")]
            avg_age = sum(avg_ages) / len(avg_ages) if avg_ages else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Consultations", total_consults)
            col2.metric("Average Patient Age", f"{avg_age:.0f} years")
            col3.metric("Most Common Symptom", "fatigue" if total_consults > 0 else "N/A")

            st.markdown("---")

            # Display consultations
            for i, consult in enumerate(reversed(st.session_state.consultation_history[-20:]), 1):
                timestamp = consult["timestamp"]
                date_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp))

                with st.expander(f"Consultation #{len(st.session_state.consultation_history) - i + 1}: {date_str}", expanded=False):
                    # Patient info
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Patient Information**")
                        st.write(f"- Age: {consult['patient'].get('age', 'N/A')} years")
                        st.write(f"- Sex: {consult['patient'].get('sex', 'N/A')}")
                        if consult['patient'].get('symptoms'):
                            st.write(f"- Symptoms: {', '.join(consult['patient']['symptoms'][:3])}")

                    with col_b:
                        st.write("**Laboratory Highlights**")
                        labs = consult['patient'].get('labs', {})
                        if labs:
                            displayed = 0
                            for key, value in list(labs.items())[:4]:  # Show first 4 labs
                                st.write(f"- {key.replace('_', ' ')}: {value}")
                                displayed += 1
                            if len(labs) > displayed:
                                st.write(f"- ... and {len(labs) - displayed} more")

                    st.markdown("---")

                    # Recommendation preview
                    st.write("**Clinical Interpretation Preview:**")
                    recommendation = consult["recommendation"]
                    preview = recommendation[:500] + "..." if len(recommendation) > 500 else recommendation
                    st.text_area("Interpretation", preview, height=150, disabled=True)

                    # Actions
                    col_c, col_d = st.columns(2)
                    with col_c:
                        if st.button(f"👁️ View Full", key=f"view_{i}"):
                            st.session_state.selected_consultation = consult
                            st.rerun()
                    with col_d:
                        st.download_button(
                            label=f"💾 Download",
                            data=recommendation,
                            file_name=f"interpretation_{date_str.replace(' ', '_').replace(':', '-')}.txt",
                            mime="text/plain",
                            key=f"download_{i}"
                        )

    with tab3:
        st.subheader("ℹ️ How MedRec AI Works")

        st.markdown("""
        ### 🎯 Purpose
        MedRec AI transforms raw laboratory results and patient complaints into **structured, patient-friendly clinical interpretations** following a strict 5-block format. The system is designed to:
        - Save physicians time on routine interpretations
        - Ensure consistent, evidence-based recommendations
        - Maintain full physician oversight and control
        - Operate 100% locally with no data leaving your machine
        
        ### 🔄 Workflow Overview
        
        **Step 1: Setup (One-Time)**
        1. Upload your **system prompt** defining the 5-block interpretation structure
        2. Upload **20+ examples** of your previous interpretations for RAG learning
        3. (Optional) Upload **clinical guidelines** (PDF/TXT) for enhanced knowledge
        4. Initialize the **5-agent system**
        
        **Step 2: Patient Consultation**
        1. Enter patient data (manual or PDF upload)
        2. System validates laboratory values for safety flags
        3. **Multi-agent collaboration begins:**
           - Primary Care Physician → Initial assessment
           - Specialist Consultant → Domain-specific insights
           - Safety Officer → Risk assessment
           - Guidelines Specialist → Evidence verification
           - Patient Communication → Plain-language translation
        4. Final interpretation generated with RAG-enhanced context
        5. Physician reviews, edits if needed, and approves
        
        ### 🤖 The 5-Agent System
        
        | Agent | Role | Function |
        |-------|------|----------|
        | **👨‍⚕️ Primary Care** | Initial Assessment | Generates first-pass clinical interpretation following 5-block structure |
        | **🔬 Specialist** | Domain Expertise | Adds specialty-specific insights (endocrinology, cardiology, etc.) |
        | **⚠️ Safety Officer** | Risk Auditor | Checks for red flags, contraindications, critical values |
        | **📚 Guidelines** | Evidence Verifier | Ensures alignment with current clinical guidelines |
        | **💬 Patient Comm** | Translator | Converts medical jargon to 8th-grade literacy level |
        
        ### 📋 The 5-Block Interpretation Structure
        
        Every interpretation follows this exact structure:
        
        1. **Systems and organs within normal range**  
           Explanation of normal parameters and what they indicate
        
        2. **Detected abnormalities and possible causes**  
           Interpretation of out-of-range values with potential causes
        
        3. **Additional comments on reported complaints**  
           Correlation of symptoms with laboratory findings
        
        4. **Recommendations for additional tests**  
           Evidence-based testing suggestions with preparation instructions
        
        5. **Recommendations for specialist consultations**  
           Appropriate specialist referrals based on findings
        
        ### 🔒 Privacy & Compliance
        
        - **100% Local Processing**: No data leaves your machine
        - **HIPAA Compliant**: No cloud transmission of PHI
        - **Physician Oversight**: You review and approve all interpretations
        - **Soft Formulations**: No direct patient addressing, no imperatives
        - **Evidence-Based**: Recommendations aligned with international guidelines
        
        ### 💡 Key Features
        
        - **RAG-Powered**: Learns from your previous interpretations
        - **PDF Processing**: Automatic extraction of lab values from reports
        - **Safety Validation**: Automatic flagging of critical values
        - **Real-Time Visualization**: Watch agents collaborate in real-time
        - **Consultation History**: Track all interpretations with search
        - **Export Capabilities**: Download interpretations as TXT files
        
        ### 🚀 Getting Started
        
        1. **Install LM Studio** and load MedGemma model
        2. **Prepare your examples** (20+ JSON interpretations)
        3. **Upload system prompt** with 5-block structure
        4. **Initialize agents** and start interpreting!
        
        For detailed setup instructions, see the README.md file in the repository.
        """)

else:
    # Welcome screen when not initialized
    st.info("""
    👈 **Complete the 4-step setup in the sidebar to begin generating clinical interpretations:**
    
    ### 📋 Setup Checklist
    
    1. **Upload System Prompt** (Required)
       - Define the 5-block interpretation structure
       - Specify clinical standards and formatting rules
       - Include soft formulation requirements
    
    2. **Upload Physician Examples** (Required, min 20)
       - JSON file with your previous interpretations
       - Each example: patient context + 5-block interpretation
       - Powers RAG for context-aware recommendations
    
    3. **Upload Knowledge Base** (Optional)
       - Clinical guidelines in PDF/TXT format
       - Enhances evidence-based recommendations
       - Supports specialty-specific protocols
    
    4. **Initialize Multi-Agent System** (Required)
       - Creates 5 specialized medical agents
       - Loads RAG examples into vector database
       - Prepares system for patient consultations
    
    ---
    
    ### ✅ System Requirements
    
    - **LM Studio** running with MedGemma model loaded
    - **Python 3.9+** with required dependencies installed
    - **Minimum 8GB RAM** for smooth operation
    - **Local machine** (no internet required after setup)
    
    ### 🎯 What You'll Get
    
    - **Time Savings**: Generate interpretations 5-10x faster
    - **Consistency**: Standardized 5-block structure every time
    - **Safety**: Automatic red flag detection and validation
    - **Privacy**: 100% local processing, no data transmission
    - **Control**: Full physician oversight and editing capability
    
    ### 📚 Resources
    
    - **System Prompt Template**: See example in sidebar
    - **JSON Examples Format**: See example in sidebar  
    - **Full Documentation**: README.md in repository
    - **Support**: Contact development team for assistance
    """)

    # Show example system prompt structure
    with st.expander("📄 Example System Prompt Structure (5-Block Format)", expanded=False):
        st.markdown("""
        **Copy and customize this template for your system prompt:**
        """)
        st.code("""
You are an experienced remote physician practicing evidence-based medicine. You receive laboratory test results and a patient questionnaire. Your task is to generate a patient-friendly clinical interpretation following the strict 5-block algorithm below.

### Purpose
Explain laboratory results and complaints in simple, non-medical language and provide a clear action plan: recommended additional tests and specialist consultations.

### Structure (5 mandatory blocks in order)

1. **Systems and organs within normal range**
   Interpret all normal parameters, explaining which organ/system functions they reflect.

2. **Detected abnormalities and possible causes**
   Interpret all out-of-range values. For each: define the parameter, list common causes, and correlate with questionnaire data.

3. **Additional comments on reported complaints**
   Link symptoms from the questionnaire to possible causes, especially if supported by laboratory abnormalities.

4. **Recommendations for additional tests and preparation**
   Suggest only basic, evidence-based tests per international clinical guidelines. Always include preparation instructions if relevant. Add: "Indications for this test should be determined by a physician during an in-person consultation."

5. **Recommendations for specialist consultations**
   Recommend relevant specialists based on symptoms and abnormalities.

### Critical Rules (Non-Negotiable)

SOFT FORMULATIONS:
- NEVER use personal pronouns ("your," "you," "yourself")
- NEVER use imperative verbs ("take," "visit," "do")
- NEVER use "recommended" alone. Always soften: "may be useful," "is often considered"
- NEVER use the word "diagnosis" or refer to "the patient"
- Keep all statements impersonal: "The analysis shows..." not "Your analysis shows..."

TREATMENT RESTRICTIONS:
- NEVER suggest treatment (lifestyle changes, medications)
- EXCEPTION 1: For iron deficiency - include standardized text about supplementation phases, dietary sources, absorption tips
- EXCEPTION 2: For vitamin D deficiency - include evidence-based correction approaches, monitoring advice, emphasize physician supervision

RED FLAGS:
If laboratory results suggest dangerous conditions (hemoglobin <90 g/L, critical electrolytes, etc.), explicitly recommend urgent specialist consultation.

### Formatting
- Follow the 5-block structure exactly
- Use plain text only (no bold, italics, asterisks)
- Use paragraph breaks for readability
- Maintain impersonal, patient-friendly language throughout
        """, language="text")

    # Show example JSON structure
    with st.expander("📄 Example Physician Examples JSON Structure", expanded=False):
        st.markdown("""
        **Your JSON file should contain an array of examples like this:**
        """)
        st.code("""
[
  {
    "example_id": "case_001",
    "patient_context": {
      "age": 52,
      "sex": "Male",
      "symptoms": ["fatigue", "increased thirst", "blurred vision"],
      "labs": {
        "fasting_glucose_mg_dl": 134,
        "hba1c_percent": 6.9,
        "total_cholesterol_mg_dl": 248,
        "ldl_cholesterol_mg_dl": 168
      },
      "questionnaire": "BMI 31.2 kg/m². Father has type 2 diabetes. Sedentary lifestyle."
    },
    "physician_recommendation": {
      "block_1_normal_systems": "Normal kidney function (creatinine 0.9 mg/dL, eGFR 90 mL/min/1.73m²) indicates adequate renal filtration. Normal liver enzymes suggest preserved hepatic function...",
      "block_2_abnormalities": "Elevated fasting glucose to 134 mg/dL and hemoglobin A1c to 6.9% meet criteria for type 2 diabetes mellitus per ADA guidelines. Elevated LDL cholesterol (168 mg/dL) significantly increases cardiovascular risk...",
      "block_3_symptom_correlation": "Fatigue and increased thirst represent classic symptoms of hyperglycemia, resulting from osmotic diuresis. Blurred vision may reflect temporary lens swelling due to glucose fluctuations...",
      "block_4_additional_tests": "Dilated eye examination may be appropriate to exclude diabetic retinopathy. Preparation includes mydriatic drops causing temporary driving impairment. Urine albumin-to-creatinine ratio screening may be useful for early nephropathy detection...",
      "block_5_specialist_consultations": "Endocrinology consultation may be appropriate for comprehensive diabetes management strategy. Cardiology consultation may be considered given elevated cardiovascular risk profile..."
    }
  },
  ... (19 more examples)
]
        """, language="json")

    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        ### 5-Minute Setup
        
        **Step 1: Install LM Studio** (2 minutes)
        1. Download from https://lmstudio.ai/
        2. Install and open the application
        3. Search for "medgemma-1.5-4b-it" in model search
        4. Download and load the model
        5. Click "Server" → "Start Server" (port 1234)
        
        **Step 2: Prepare Your Data** (2 minutes)
        1. Create `system_prompt.txt` with the 5-block structure above
        2. Prepare `clinical_examples.json` with 20+ of your previous interpretations
        3. (Optional) Gather clinical guidelines as PDF/TXT files
        
        **Step 3: Launch MedRec AI** (1 minute)
        1. Run `streamlit run app.py`
        2. Open http://localhost:8501 in your browser
        3. Upload your files in the sidebar
        4. Click "Initialize Multi-Agent System"
        5. Start generating interpretations!
        
        ### Troubleshooting
        
        **LM Studio not connecting?**
        - Check that server is running on port 1234
        - Verify firewall isn't blocking localhost
        - Try restarting LM Studio
        
        **Agents won't initialize?**
        - Check that system prompt is uploaded
        - Verify JSON examples file has at least 20 entries
        - Ensure all required Python packages are installed
        
        **PDF extraction failing?**
        - Try a different PDF file
        - Ensure PDF contains machine-readable text (not scanned images)
        - Enter values manually as workaround
        
        ### Need Help?
        
        - Check the README.md for detailed documentation
        - Review example files in the `data/sample_prompts/` directory
        - Contact the development team for support
        """)

# === FOOTER ===
st.markdown("---")
st.caption("""
🔒 100% local processing | No data leaves your machine | Powered by MedGemma via LM Studio | 
Multi-agent clinical decision support system v1.0
""")

# Add version info in sidebar footer
with st.sidebar:
    st.markdown("---")
    st.caption("""
    **MedRec AI v1.0**  
    Multi-Agent Clinical Decision Support  
    © 2026 All rights reserved
    """)