import streamlit as st
import joblib
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from collections import Counter
import nltk
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

# NLTK WordNet letöltése
nltk.download('wordnet')

# SentenceTransformer modell betöltése
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = load_sentence_transformer()

# Page Configuration
st.set_page_config(page_title="VetAlert", layout="centered")

# Load external CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("deployment/style.css")

def apply_inline_styles():
    css = """
    body {
        color: #ffffff;
    }
    .stButton>button {
        color: white;
        background-color: #007bff;
    }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

apply_inline_styles()

# Similarity Functions
def sequence_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def semantic_similarity(a, b, model):
    embeddings = model.encode([a.lower(), b.lower()], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()

def combined_similarity(a, b, model, seq_weight=0.4, sem_weight=0.6):
    seq_sim = sequence_similarity(a, b)
    sem_sim = semantic_similarity(a, b, model)
    return seq_weight * seq_sim + sem_weight * sem_sim

# Label Retrieval Function
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

# Szinonimakereső függvény
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            synonyms.add(synonym)
    return list(synonyms) if synonyms else [word.lower()]

# Data and Model Loading
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
    knn = joblib.load("deployment/knn_model.pkl")
    return linear, rbf, knn

@st.cache_data
def load_label_encodings():
    animal = pd.read_csv("deployment/label_encoding_Animal.csv")
    group = pd.read_csv("deployment/label_encoding_AnimalGroup.csv")
    species = pd.read_csv("deployment/label_encoding_Species.csv")
    return animal, group, species

@st.cache_data
def load_cluster_data():
    animal_clusters = pd.read_csv("deployment/cluster_distribution_by_animal.csv")
    group_clusters = pd.read_csv("deployment/cluster_distribution_by_animalgroup.csv")
    species_clusters = pd.read_csv("deployment/cluster_distribution_by_species.csv")
    return animal_clusters, group_clusters, species_clusters

linear_model, rbf_model, knn_model = load_models()
word_map = load_mapping()
animal_enc, animal_group_enc, species_enc = load_label_encodings()
animal_clusters, group_clusters, species_clusters = load_cluster_data()

# Symptom Processing
def process_symptoms_to_cluster_ids(symptoms, word_map, sentence_model, similarity_threshold=0.75):
    matched_symptoms = []
    word_map.columns = word_map.columns.str.strip()

    try:
        for symptom in symptoms:
            symptom = symptom.strip().lower()
            max_similarity = 0
            best_match_cluster_id = None

            for _, row in word_map.iterrows():
                for col in ['original', 'corrected', 'canonical']:
                    if pd.isna(row[col]):
                        continue
                    sim = combined_similarity(symptom, str(row[col]), sentence_model)
                    if sim >= similarity_threshold and sim > max_similarity:
                        max_similarity = sim
                        best_match_cluster_id = row['cluster_id']

            if best_match_cluster_id is None:
                synonyms = get_synonyms(symptom)
                for synonym in synonyms:
                    for _, row in word_map.iterrows():
                        for col in ['original', 'corrected', 'canonical']:
                            if pd.isna(row[col]):
                                continue
                            sim = combined_similarity(synonym, str(row[col]), sentence_model)
                            if sim >= similarity_threshold and sim > max_similarity:
                                max_similarity = sim
                                best_match_cluster_id = row['cluster_id']

            matched_symptoms.append(best_match_cluster_id if best_match_cluster_id is not None else -1)
        return matched_symptoms
    except Exception as e:
        st.error(f"Error processing symptoms: {str(e)}")
        return None

def check_symptoms_match(symptoms, word_map, sentence_model, threshold=0.75):
    invalid_indices = []
    word_map.columns = word_map.columns.str.strip()

    for i, symptom in enumerate(symptoms):
        symptom = symptom.strip().lower()
        max_similarity = 0

        for _, row in word_map.iterrows():
            for col in ['original', 'corrected', 'canonical']:
                if pd.isna(row[col]):
                    continue
                sim = combined_similarity(symptom, str(row[col]), sentence_model)
                if sim > max_similarity:
                    max_similarity = sim

        if max_similarity < threshold:
            synonyms = get_synonyms(symptom)
            for synonym in synonyms:
                for _, row in word_map.iterrows():
                    for col in ['original', 'corrected', 'canonical']:
                        if pd.isna(row[col]):
                            continue
                        sim = combined_similarity(synonym, str(row[col]), sentence_model)
                        if sim > max_similarity:
                            max_similarity = sim

        if max_similarity < threshold:
            invalid_indices.append(i+1)
    return invalid_indices

# Check ClusterID in CSV Files
def check_cluster_ids_in_csv(cluster_ids, animal_order, animal_group, animal_breed):
    missing_clusters = []
    valid_cluster_ids = [cid for cid in cluster_ids if cid != -1]

    if animal_breed and animal_breed not in ["Unknown", "Select..."]:
        species_data = species_clusters[species_clusters['Species'].str.lower() == animal_breed.lower()]
        species_cluster_ids = species_data['ClusterID'].tolist()
        missing_clusters = [cid for cid in valid_cluster_ids if cid not in species_cluster_ids]
        if missing_clusters:
            return missing_clusters, "species"

    if animal_order and animal_order not in ["Unknown", "Select..."]:
        animal_data = animal_clusters[animal_clusters['Animal'].str.lower() == animal_order.lower()]
        animal_cluster_ids = animal_data['ClusterID'].tolist()
        missing_clusters = [cid for cid in valid_cluster_ids if cid not in animal_cluster_ids]
        if missing_clusters:
            return missing_clusters, "animal"

    if animal_group and animal_group != "Select...":
        group_data = group_clusters[group_clusters['AnimalGroup'].str.lower() == animal_group.lower()]
        group_cluster_ids = group_data['ClusterID'].tolist()
        missing_clusters = [cid for cid in valid_cluster_ids if cid not in group_cluster_ids]
        if missing_clusters:
            return missing_clusters, "group"

    return [], None

# UI Components
st.image("deployment/banner animal.png", use_container_width=True)
st.markdown("""
<header style="border: 3px solid #00ff5b; padding: 15px; border-radius: 10px; background-color: #00ff5b;">
    <h1 style="text-align: center; color: #0e1711;">VetAlert</h1>
    <h3 style="text-align: center; color: #0e1711;">Your AI-Powered Animal Condition Classification Tool</h3>
</header>
<div style="padding-top: 20px;"></div>
<header style="border: 3px solid #00ff5b; padding: 15px; border-radius: 10px; background-color: #0e1711;">
    <h5 style="text-align: center; color: #00ff5b;">This is a web-based tool designed to assist in identifying potential health conditions in animals using basic animal information and observed symptoms.</h5>
    <h5 style="text-align: center; color: white;"><i>Please note that this tool is not a substitute for professional veterinary advice. Always consult a veterinarian for accurate diagnosis and treatment.</i></h5>
</header>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    label {
        color: #003300 !important;
        font-size: 16px !important;
        font-weight: bold !important;
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
    elif animal_order == "Rodent":
        animal_breed = st.selectbox("Breed", ["Select...", "Rabbit", "Hamster"])
    elif animal_order in ["Other", "Elephant", "Monkey"]:
        animal_order = 'Unknown' if animal_order == "Other" else animal_order
        animal_breed = "Unknown"

elif animal_group == "Bird":
    animal_order = st.selectbox("Bird Type", ["Select...", "Fowl", "Other birds"])
    if animal_order == "Fowl":
        animal_breed = st.selectbox("Breed", ["Select...", "Chicken", "Duck", "Other"])
        if animal_breed == "Other":
            animal_breed = "Unknown"
    if animal_order == "Other birds":
        animal_breed = "Unknown"

elif animal_group == "Reptile":
    animal_order = st.selectbox("Reptile Type", ["Select...", "Turtle", "Snake"])
    animal_breed = "Unknown"

st.markdown("""
<style>
.stButton > button {
    background-color: #18291e;
    color: #00ff5b;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.5em 1em;
    border: 3px solid #00ff5b;
}
.stButton > button:hover {
    background-color: #18291e;
    color: #68fc9d;
    font-weight: bold;
    border: 3px solid #68fc9d;
}
</style>
""", unsafe_allow_html=True)

# Symptoms Input
st.subheader("Symptoms")
symptoms = [st.text_input(f"Symptom {i+1}", placeholder="Enter symptom here") for i in range(5)]

# Initialize session state
if 'proceed_diagnosis' not in st.session_state:
    st.session_state.proceed_diagnosis = None

# Diagnose Logic
if st.button("Diagnose"):
    if animal_group == "Select..." or animal_order == "Select..." or (animal_breed == "Select..." and animal_breed is not None):
        st.error("Please complete all animal selections.")
    elif any(s.strip() == '' for s in symptoms):
        st.error("Please fill in all symptom fields.")
    else:
        invalid_symptoms = check_symptoms_match(symptoms, word_map, sentence_model)
        if invalid_symptoms:
            for i in invalid_symptoms:
                st.error(f"Symptom {i} is not recognized. Please check the spelling or try a different term.")
        else:
            cluster_ids = process_symptoms_to_cluster_ids(symptoms, word_map, sentence_model)
            if cluster_ids is None:
                st.error("Symptom processing failed.")
            else:
                missing_clusters, level = check_cluster_ids_in_csv(cluster_ids, animal_order, animal_group, animal_breed)
                if missing_clusters:
                    st.warning(f"Warning: The following symptom Cluster IDs {missing_clusters} are not associated with the selected {level}. Do you want to proceed with the diagnosis?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes"):
                            st.session_state.proceed_diagnosis = True
                    with col2:
                        if st.button("No"):
                            st.session_state.proceed_diagnosis = False
                else:
                    st.session_state.proceed_diagnosis = True

# Perform Diagnosis
if st.session_state.proceed_diagnosis is True:
    enc_animal = get_label_from_df(animal_enc, str(animal_order))
    enc_group = get_label_from_df(animal_group_enc, str(animal_group))
    enc_species = get_label_from_df(species_enc, str(animal_breed))

    if None in [enc_animal, enc_group, enc_species]:
        st.error("Unable to encode animal information. Please verify the input data.")
    else:
        cluster_ids = process_symptoms_to_cluster_ids(symptoms, word_map, sentence_model)
        if cluster_ids is None:
            st.error("Symptom processing failed.")
        else:
            model_input = [enc_animal, enc_group, enc_species] + cluster_ids
            try:
                proba_linear = linear_model.predict_proba([model_input])[0]
                proba_rbf = rbf_model.predict_proba([model_input])[0]
                proba_knn = knn_model.predict_proba([model_input])[0]

                pred_linear = np.argmax(proba_linear)
                pred_rbf = np.argmax(proba_rbf)
                pred_knn = np.argmax(proba_knn)

                weights = {'linear': 0.4, 'rbf': 0.3, 'knn': 0.3}
                vote_counts = Counter()
                vote_counts[pred_linear] += weights['linear']
                vote_counts[pred_rbf] += weights['rbf']
                vote_counts[pred_knn] += weights['knn']

                prediction = max(vote_counts.items(), key=lambda x: (x[1], x[0] == pred_linear))[0]

                avg_proba = (proba_linear[prediction] + proba_rbf[prediction] + proba_knn[prediction]) / 3
                confidence = avg_proba * 100

                if prediction == 1:
                    st.error(f"⚠️☠️🚨 Dangerous condition      Confidence: {confidence:.2f}%")
                    if confidence > 90:
                        st.warning("⚠️🚨💉 Immediate veterinary attention is recommended!")
                    elif confidence > 80:
                        st.warning("⚠️🚨💉 High confidence in dangerous condition. Please consult a vet.")
                    elif confidence > 70:
                        st.warning("⚠️🚨💉 Moderate confidence in dangerous condition. Please consult a vet.")
                    elif confidence > 60:
                        st.warning("⚠️🚨💉 Low confidence in dangerous condition. Please consult a vet.")
                    else:
                        st.warning("⚠️🚨💉 Very low confidence in dangerous condition. Please consult a vet.")
                elif prediction == 0:
                    st.success(f"🩺👩🏻‍⚕️✅ Non-dangerous condition        Confidence: {confidence:.2f}%")
                    if confidence > 90:
                        st.success("🩺👩🏻‍⚕️✅ High confidence in non-dangerous condition.")
                    elif confidence > 80:
                        st.success("🩺👩🏻‍⚕️✅ Moderate confidence in non-dangerous condition.")
                    elif confidence > 70:
                        st.success("🩺👩🏻‍⚕️✅ Low confidence in non-dangerous condition.")
                    elif confidence > 40:
                        st.success("🩺👩🏻‍⚕️✅ Very low confidence in non-dangerous condition.")
                    else:
                        st.success("🩺👩🏻‍⚕️✅ Extremely low confidence in non-dangerous condition.")
                else:
                    st.info(f"😵‍💫 Insecure condition        Confidence: {confidence:.2f}%")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
elif st.session_state.proceed_diagnosis is False:
    st.error("Diagnosis cannot be continued due to missing Cluster IDs.")