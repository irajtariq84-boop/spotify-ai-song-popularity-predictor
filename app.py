import streamlit as st
import pandas as pd
import joblib


# ============================================
# LOAD MODELS
# ============================================

regression_model = joblib.load(
    "spotify_regression_model.pkl"
)

classification_model = joblib.load(
    "spotify_classifier_model.pkl"
)

scaler = joblib.load("scaler.pkl")


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(

    page_title="Spotify AI Predictor",
    page_icon="🎧",
    layout="centered"

)

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #121212,
            #1DB954
        );
        color: white;
    }

    /* Title */
    h1 {
        color: white;
        text-align: center;
        font-size: 45px;
    }

    /* Subheaders */
    h2, h3 {
        color: #F5F5F5;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 12px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #17a74a;
        color: white;
    }

    /* Input boxes */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 5px;
    }

    /* Prediction cards */
    .prediction-box {
        background-color: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
    }

    </style>
    """,

    unsafe_allow_html=True
)


# ============================================
# TITLE
# ============================================

st.title("🎧 Spotify Song Popularity Predictor")

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/8/84/Spotify_icon.svg",
    width=100
)

st.write(
    "Predict Spotify song popularity using AI."
)


# ============================================
# USER INPUTS
# ============================================

artist_popularity = st.slider(
    "Artist Popularity",
    0,
    100,
    50
)

artist_followers = st.number_input(
    "Artist Followers",
    min_value=0,
    value=100000
)

track_duration_ms = st.number_input(
    "Track Duration (ms)",
    min_value=0,
    value=200000
)

album_total_tracks = st.number_input(
    "Album Total Tracks",
    min_value=1,
    value=10
)

track_number = st.number_input(
    "Track Number",
    min_value=1,
    value=1
)

explicit = st.selectbox(
    "Explicit Content",
    [0, 1]
)


# ============================================
# PREDICTION BUTTON
# ============================================

if st.button("Predict Popularity"):

    # ============================================
    # CREATE INPUT DATAFRAME
    # ============================================

    input_data = pd.DataFrame({

    'artist_popularity': [artist_popularity],

    'artist_followers': [artist_followers],

    'album_total_tracks': [album_total_tracks],

    'track_number': [track_number],

    'track_duration_ms': [track_duration_ms],

    'explicit': [explicit]

    })


    # SCALE FEATURES
    scaled_columns = [

    'artist_popularity',
    'artist_followers',
    'album_total_tracks',
    'track_number',
    'track_duration_ms'

]

    input_data[scaled_columns] = scaler.transform(
        input_data[scaled_columns]
    )


    # REGRESSION PREDICTION
    popularity_score = regression_model.predict(
        input_data
    )[0]


    # CLASSIFICATION PREDICTION
    popularity_category = classification_model.predict(
        input_data
    )[0]


    # ============================================
    # OUTPUTS
    # ============================================

    st.success("Prediction Completed ✅")

    st.markdown(f"""

    <div class="prediction-box">

    <h2>🎵 Predicted Popularity Score</h2>

    <h1>{round(popularity_score, 2)}</h1>

    </div>

    """, unsafe_allow_html=True)


    st.markdown(f"""

    <div class="prediction-box">

    <h2>🔥 Popularity Category</h2>

    <h1>{popularity_category}</h1>

    </div>

    """, unsafe_allow_html=True)