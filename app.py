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


# ============================================
# TITLE
# ============================================

st.title("🎧 Spotify Song Popularity Predictor")

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

    input_data = pd.DataFrame([[

        artist_popularity,
        artist_followers,
        track_duration_ms,
        album_total_tracks,
        track_number,
        explicit

    ]], columns=[

        'artist_popularity',
        'artist_followers',
        'track_duration_ms',
        'album_total_tracks',
        'track_number',
        'explicit'

    ])


    # SCALE FEATURES
    scaled_columns = [

        'artist_popularity',
        'artist_followers',
        'track_duration_ms',
        'album_total_tracks',
        'track_number'

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

    st.subheader("🎵 Predicted Popularity Score")

    st.write(round(popularity_score, 2))

    st.subheader("🔥 Popularity Category")

    st.write(popularity_category)