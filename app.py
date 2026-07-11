import requests
import streamlit as st
import numpy as np
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:latest"
SECOND_MODEL = "qwen2.5:3b"

st.set_page_config(
    page_title="AI Automated Triage Assistant",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 AI Automated Triage System Assistant")
st.caption("Input patient data from the local dataset, review the patient profile, and ask the local AI model for a triage recommendation.")

if "triage_history" not in st.session_state:
    st.session_state.triage_history = {}


@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n = 2500

    arrival_modes = ["Ambulance", "Walk-in", "Wheelchair"]
    triage_levels = [1, 2, 3]

    df = pd.DataFrame(
        {
            "patient_id": range(1, n + 1),
            "age": np.random.randint(18, 90, n),
            "heart_rate": np.random.normal(90, 20, n).astype(int),
            "systolic_blood_pressure": np.random.normal(120, 20, n).astype(int),
            "oxygen_saturation": np.round(np.random.normal(96, 4, n), 1),
            "body_temperature": np.round(np.random.normal(37, 1.2, n), 1),
            "pain_level": np.random.randint(1, 11, n),
            "chronic_disease_count": np.random.poisson(1.5, n),
            "previous_er_visits": np.random.poisson(2, n),
            "arrival_mode": np.random.choice(arrival_modes, n, p=[0.30, 0.50, 0.20]),
            "triage_level": np.random.choice(triage_levels, n, p=[0.20, 0.50, 0.30]),
        }
    )

    return df


def get_patient_record(dataset: pd.DataFrame, patient_id: int):
    patient_record = dataset[dataset["patient_id"] == patient_id]
    if patient_record.empty:
        return None
    return patient_record.iloc[0]


def build_patient_context(patient_record: pd.Series) -> str:
    return f"""
Patient Dataset Snapshot
- Patient ID: {int(patient_record['patient_id'])}
- Age: {int(patient_record['age'])}
- Heart Rate: {int(patient_record['heart_rate'])}
- Systolic Blood Pressure: {int(patient_record['systolic_blood_pressure'])}
- Oxygen Saturation: {float(patient_record['oxygen_saturation'])}
- Body Temperature: {float(patient_record['body_temperature'])}
- Pain Level: {int(patient_record['pain_level'])}
- Chronic Disease Count: {int(patient_record['chronic_disease_count'])}
- Previous ER Visits: {int(patient_record['previous_er_visits'])}
- Arrival Mode: {patient_record['arrival_mode']}
- Current Triage Level (dataset label): {int(patient_record['triage_level'])}
"""


def call_ollama(prompt: str, model_name: str, temp: float) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temp,
            "num_predict": 512,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json().get("response", "")


with st.sidebar:
    st.header("⚙️ Configuration")
    model = st.selectbox("LLM Model", [DEFAULT_MODEL, SECOND_MODEL])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    st.divider()
    st.success(f"Model: {model}")


patient_dataset = generate_dataset()

col1, col2 = st.columns([1.15, 0.85])

with col1:
    st.subheader("1) Patient Input Form")

    with st.form("triage_assessment_form"):
        patient_id = st.number_input(
            "Patient ID",
            min_value=1,
            max_value=len(patient_dataset),
            value=1,
            step=1,
        )
        patient_note = st.text_area(
            "Patient symptoms / clinical note",
            height=180,
            placeholder="Example: 58-year-old patient with chest pain, shortness of breath, dizziness, and low oxygen saturation.",
        )
        submit = st.form_submit_button("🚀 Generate Triage Assessment")

    patient_record = get_patient_record(patient_dataset, int(patient_id))

    if patient_record is None:
        st.warning("Patient ID not found in the generated dataset.")
    else:
        st.success(f"Patient profile loaded for ID: {int(patient_id)}")
        st.dataframe(patient_record.to_frame().T, use_container_width=True)

with col2:
    st.subheader("2) Historical Chat by Patient ID")
    history_key = f"patient_{int(patient_id)}"
    history = st.session_state.triage_history.get(history_key, [])

    if history:
        for item in history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
    else:
        st.info("No previous triage chat for this patient yet. Submit the form to create one.")

if submit:
    if patient_record is None:
        st.error("Please use a valid patient ID from the generated dataset.")
    else:
        context = build_patient_context(patient_record)
        note = patient_note.strip() or "No additional clinical notes were provided."

        full_prompt = f"""
You are an AI triage assistant for a hospital emergency intake workflow.

Based on the patient dataset and the clinical note, classify the patient into a triage level and explain the recommendation.

Return the answer in Bahasa Indonesia with:
1. Triage Level
2. Reasoning
3. Key warning signs
4. Recommended action

{context}

Clinical note:
{note}
"""

        with st.spinner("Analyzing patient condition with Ollama..."):
            try:
                result = call_ollama(full_prompt, model, temperature)
                history = st.session_state.triage_history.setdefault(history_key, [])
                history.append({"role": "user", "content": note})
                history.append({"role": "assistant", "content": result})
                st.subheader("3) AI Triage Output")
                st.markdown(result)
            except requests.exceptions.ConnectionError:
                st.error("Tidak bisa terhubung ke Ollama. Pastikan Ollama sudah jalan: `ollama serve`")
            except requests.exceptions.HTTPError as e:
                st.error(f"Ollama HTTP error: {e}")
                st.info(f"Cek apakah model sudah di-pull: `ollama pull {model}`")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.markdown(
    """
### Concept shown
- **Streamlit** = web UI for the triage assistant
- **Ollama** = local model runtime
- **Dataset lookup** = patient record retrieval based on patient ID
- **Per-patient history** = past triage conversation can be reused by the same patient ID
"""
)