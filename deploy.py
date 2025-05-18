import streamlit as st
import joblib
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# ----- Page Configuration -----
st.set_page_config(
    page_title="VetAlert",
    layout="centered",
)

# ----- Load external CSS -----
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("deployment/style.css")  # vagy "styles/style.css" ha almappában van

def apply_inline_styles():
    css = """
    body {
        font-color: #ffffff;
    }
    .stButton>button {
        color: white;
        background-color: #007bff;
    }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    
apply_inline_styles()

# ----- Similarity Function -----
def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ----- Label Retrieval Function -----
def get_label_from_df(df, input_value):
    try:
        input_value = input_value.strip().lower()
        match = df[df['animal'].str.lower() == input_value]
        if not match.empty:
            return int(match.iloc[0]['label'])
        return None
    except Exception as e:
        st.error(f"Error in label encoding: {str(e)}")
        return None

# ----- Data and Model Loading -----
@st.cache_data
def load_mapping():
    return pd.read_csv(
        "deployment/symptom_word_mapping.csv",
        usecols=['original', 'corrected', 'canonical', 'cluster_id'],
        encoding='utf-8'
    )

@st.cache_resource
def load_models():
    linear = joblib.load("deployment/svmlinear_model.pkl")
    rbf = joblib.load("deployment/svmrbf_model.pkl")
    return linear, rbf

linear_model, rbf_model = load_models()

@st.cache_data
def load_label_encodings():
    animal = pd.read_csv("deployment/label_encoding_Animal.csv")
    group = pd.read_csv("deployment/label_encoding_AnimalGroup.csv")
    species = pd.read_csv("deployment/label_encoding_Species.csv")
    return animal, group, species

word_map = load_mapping()
animal_enc, animal_group_enc, species_enc = load_label_encodings()

# ----- Symptom Processing -----
def process_symptoms_to_cluster_ids(symptoms, word_map):
    matched_symptoms = []
    word_map.columns = word_map.columns.str.strip()

    try:
        for symptom in symptoms:
            symptom = symptom.strip()
            max_similarity = 0
            best_match = None

            for _, row in word_map.iterrows():
                for col in ['original', 'corrected', 'canonical']:
                    if pd.isna(row[col]):
                        continue
                    sim = similar(symptom, str(row[col]))
                    if sim >= 0.75 and sim > max_similarity:
                        max_similarity = sim
                        best_match = row['cluster_id']

            matched_symptoms.append(best_match if best_match is not None else -1)
        return matched_symptoms
    except Exception as e:
        st.error(f"Error processing symptoms: {str(e)}")
        return None

def check_symptoms_match(symptoms, word_map, threshold=0.75):
    invalid_indices = []
    word_map.columns = word_map.columns.str.strip()

    for i, symptom in enumerate(symptoms):
        symptom = symptom.strip()
        max_similarity = 0
        for _, row in word_map.iterrows():
            for col in ['original', 'corrected', 'canonical']:
                if pd.isna(row[col]):
                    continue
                sim = similar(symptom, str(row[col]))
                if sim > max_similarity:
                    max_similarity = sim
        if max_similarity < threshold:
            invalid_indices.append(i+1)  # 1-based index a felhasználónak
    return invalid_indices


# ----- UI Components -----
st.image("deployment/banner animal.png", use_container_width=True)
st.markdown("""
<header>
    <h1 style="text-align: center; color: ##1a6116;">VetAlert</h1>
    <h3 style="text-align: center; color: ##1f6e1b;">Your AI-Powered Animal Condition Classification Tool</h3>
</header>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        label {
            color: ##003300 !important;
            font-size: 16px; !important;
            font-weight: bold; !important;
        }
    </style>
""", unsafe_allow_html=True)


st.subheader("Animal Information")
animal_group = st.selectbox("Animal Group", ["Select...", "Mammal", "Bird", "Reptile"])
animal_order = None
animal_breed = None

if animal_group == "Mammal":
    animal_order = st.selectbox("Animal Subgroup", ["Select...", "Canid", "Cat-like", "Even-toed ungulates", "Odd-toed ungulates", "Elephant", "Rodent", "Monkey", "Other"])
    if animal_order == "Canid":
        animal_breed = st.selectbox("Breed", ["Select...", "Dog", "Wolf", "Fox", "Hyena"])
    elif animal_order == "Cat-like":
        animal_breed = st.selectbox("Breed", ["Select...", "Cat", "Tiger", "Lion"])
    elif animal_order == "Even-toed ungulates":
        animal_breed = st.selectbox("Breed", ["Select...", "Cattle", "Sheep", "Goat", "Buffalo", "Deer", "Pig"])
    elif animal_order == "Odd-toed ungulates":
        animal_breed = st.selectbox("Breed", ["Select...", "Horse", "Donkey", "Mule"])
        
    elif animal_order == "Roddent":
        animal_breed = st.selectbox("Breed", ["Select...", "Rabbit", "Hamster"])
    elif animal_order == "Other":
        animal_breed = "Unknown"
 
elif animal_group == "Bird":
    animal_order = st.selectbox("Bird Type", ["Select...", "Fowl", "Other birds"])
    if animal_order == "Fowl":
        animal_breed = st.selectbox("Breed", ["Select...", "Chicken", "Duck", "Unknown"])
    else:
        animal_breed="Unknown"

elif animal_group == "Reptile":
    animal_order = st.selectbox("Reptile Type", ["Select...", "Turtle", "Snake"])
    animal_breed = "Unknown"



# Symptoms Input
st.subheader("Symptoms")
symptoms = [st.text_input(f"Symptom {i+1}", placeholder="Enter symptom here") for i in range(5)]

if st.button("Diagnose"):

    if animal_group == "Select..." or animal_order == "Select..." or animal_breed == "Select...":
        st.error("Please complete all animal selections.")
    elif any(s.strip() == '' for s in symptoms):
        st.error("Please fill in all symptom fields.")
    else:
        # Ellenőrzés, hogy minden symptom legalább 75%-ban passzol-e
        invalid_symptoms = check_symptoms_match(symptoms, word_map)

        if invalid_symptoms:
            for i in invalid_symptoms:
                st.error(f"Symptom {i} is not recognized. Please check the spelling or try a different term.")
        else:
            # Ha minden ok, mehet a kód többi része (animal encoding, symptom cluster ids, predikció)
            enc_animal = get_label_from_df(animal_enc, str(animal_order))
            enc_group = get_label_from_df(animal_group_enc, str(animal_group))
            enc_species = get_label_from_df(species_enc, str(animal_breed))

            if None in [enc_animal, enc_group, enc_species]:
                st.error("Unable to encode animal information. Please verify the input data.")
            else:
                cluster_ids = process_symptoms_to_cluster_ids(symptoms, word_map)

                if cluster_ids is None:
                    st.error("Symptom processing failed.")
                else:
                    model_input = [enc_animal, enc_group, enc_species] + cluster_ids
                    try:
                        proba_linear = linear_model.predict_proba([model_input])[0]
                        proba_rbf = rbf_model.predict_proba([model_input])[0]

                        avg_proba = (proba_linear + proba_rbf) / 2
                        prediction = np.argmax(avg_proba)
                        confidence = avg_proba[prediction] * 100

                        if prediction == 1:
                            st.error(f"🛑 Diagnosis: Dangerous condition detected. Confidence: {confidence:.2f}%")
                        elif prediction == 0:
                            st.success(f"✅ Diagnosis: Non-dangerous condition. Confidence: {confidence:.2f}%")
                        else:
                            st.info(f"ℹ️ Diagnosis: Insecure condition. Confidence: {confidence:.2f}%")

                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

