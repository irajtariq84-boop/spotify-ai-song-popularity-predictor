import streamlit as st
import pandas as pd
import joblib


# ============================================
# LOAD MODEL + SCALER
# ============================================

regression_model = joblib.load(
    "spotify_regression_model.pkl"
)

scaler = joblib.load("scaler.pkl")


# ============================================
# PAGE SETTINGS
# ============================================

st.set_page_config(
    page_title="Spotify AI Predictor",
    page_icon="🎧",
    layout="centered"
)


# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #121212,
        #1DB954
    );
    color: white;
}

h1 {
    text-align: center;
    color: white;
    font-size: 45px;
}

.stButton > button {
    background-color: #1DB954;
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton > button:hover {
    background-color: #169c46;
    color: white;
}

.prediction-box {
    background-color: rgba(255,255,255,0.1);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# TITLE
# ============================================

st.title("🎧 Spotify Song Popularity Predictor")

st.write(
    "Predict Spotify song popularity using Machine Learning."
)


# ============================================
# INPUTS
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
# PREDICT BUTTON
# ============================================

if st.button("Predict Popularity"):

    # dataframe for scaler
    scale_df = pd.DataFrame({

        'artist_popularity': [artist_popularity],
        'artist_followers': [artist_followers],
        'track_duration_ms': [track_duration_ms],
        'album_total_tracks': [album_total_tracks],
        'track_number': [track_number]

    })


    # scale
    scaled_values = scaler.transform(scale_df)


    # dataframe for model
    input_data = pd.DataFrame({

        'artist_popularity': [scaled_values[0][0]],
        'artist_followers': [scaled_values[0][1]],
        'album_total_tracks': [scaled_values[0][3]],
        'track_number': [scaled_values[0][4]],
        'track_duration_ms': [scaled_values[0][2]],
        'explicit': [explicit]

    })


    # prediction
    popularity_score = regression_model.predict(
        input_data
    )[0]


    # category logic
    if popularity_score >= 70:
        category = "🔥 HIT"

    elif popularity_score >= 40:
        category = "🎵 MEDIUM"

    else:
        category = "📉 LOW"


    # outputs
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

    <h1>{category}</h1>

    </div>
    """, unsafe_allow_html=True)