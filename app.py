import streamlit as st
import pandas as pd
import joblib

# 1. Load the Model
@st.cache_resource
def load_model():
    model = joblib.load('dota2_win_predictor_model.joblib')
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found! Make sure 'dota2_win_predictor_model.joblib' is in the same folder.")
    st.stop()

# 2. Extract Hero List from the Model
# The model knows exactly which columns it was trained on (e.g., 'radiant_Abaddon')
model_features = model.feature_names_in_

# We strip 'radiant_' to get the clean hero names for the dropdowns
hero_options = sorted([f.replace('radiant_', '') for f in model_features if f.startswith('radiant_')])

# 3. App Layout
st.set_page_config(page_title="Dota 2 Draft Predictor", layout="wide")
st.title("⚔️ Dota 2 Draft Win Predictor")
st.markdown("Select the 5 heroes for each team to predict the winner based on **Logistic Regression**.")

col1, col2 = st.columns(2)

# Radiant Selection
with col1:
    st.header("Radiant Team (Green)")
    r1 = st.selectbox("Radiant Hero 1", hero_options, index=0)
    r2 = st.selectbox("Radiant Hero 2", hero_options, index=1)
    r3 = st.selectbox("Radiant Hero 3", hero_options, index=2)
    r4 = st.selectbox("Radiant Hero 4", hero_options, index=3)
    r5 = st.selectbox("Radiant Hero 5", hero_options, index=4)
    radiant_heroes = [r1, r2, r3, r4, r5]

# Dire Selection
with col2:
    st.header("Dire Team (Red)")
    # We reverse the list for Dire defaults just for variety
    d1 = st.selectbox("Dire Hero 1", hero_options, index=len(hero_options)-1)
    d2 = st.selectbox("Dire Hero 2", hero_options, index=len(hero_options)-2)
    d3 = st.selectbox("Dire Hero 3", hero_options, index=len(hero_options)-3)
    d4 = st.selectbox("Dire Hero 4", hero_options, index=len(hero_options)-4)
    d5 = st.selectbox("Dire Hero 5", hero_options, index=len(hero_options)-5)
    dire_heroes = [d1, d2, d3, d4, d5]

# 4. Prediction Logic
if st.button("Predict Winner", type="primary"):
    # VALIDATION: Check for duplicates
    all_selected = radiant_heroes + dire_heroes
    if len(set(all_selected)) != 10:
        st.error("⚠️ Error: Duplicate heroes selected! A hero can only appear once in a game.")
    else:
        # CONSTRUCT INPUT VECTOR
        # 1. Initialize a dictionary with all model columns set to 0
        input_data = {feature: 0 for feature in model_features}

        # 2. Set the chosen heroes to 1
        for hero in radiant_heroes:
            feature_name = f"radiant_{hero}"
            if feature_name in input_data:
                input_data[feature_name] = 1
        
        for hero in dire_heroes:
            feature_name = f"dire_{hero}"
            if feature_name in input_data:
                input_data[feature_name] = 1

        # 3. Create DataFrame (1 row)
        df_input = pd.DataFrame([input_data])

        # 4. Get Prediction
        # predict_proba returns [[prob_loss, prob_win]]
        probs = model.predict_proba(df_input)[0]
        radiant_prob = probs[1]
        dire_prob = probs[0]

        st.divider()
        
        # 5. Display Result
        if radiant_prob > 0.5:
            st.success(f"**RADIANT VICTORY** predicted! ({radiant_prob*100:.1f}%)")
            st.progress(radiant_prob)
        else:
            st.error(f"**DIRE VICTORY** predicted! ({dire_prob*100:.1f}%)")
            st.progress(dire_prob)