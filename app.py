import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import base64

st.set_page_config(
    page_title="GreenGuard - Guava Leaf Classifier",
    page_icon="🌿",
    layout="wide"
)

def set_bg_with_theme(light_img, dark_img, dark_mode):
    img_file = dark_img if dark_mode else light_img
    with open(img_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string.decode()}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        transition: background-image 1s ease-in-out;
        font-family: 'Segoe UI', sans-serif;
    }}
    .title {{
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: {"#FFFFFF" if dark_mode else "#1B4332"};
        text-shadow: 2px 2px 8px {"#000000" if dark_mode else "#FFFFFF"};
    }}
    .subheader {{
        text-align: center;
        font-size: 1.2em;
        color: {"#EEEEEE" if dark_mode else "#1B4332"};
        font-weight: bold;
        text-shadow: 1px 1px 4px {"#000000" if dark_mode else "#FFFFFF"};
    }}
    .upload-box {{
        background: rgba(255, 255, 255, 0.7);
        border-radius: 15px;
        padding: 15px;
        margin: 10px auto;
        max-width: 90%;
        width: 450px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        color: black;
        font-weight: bold;
    }}
    .result-card {{
        background: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        color: {"#000000" if dark_mode else "#000000"};
        font-weight: bold;
        transition: transform 0.3s ease;
        max-width: 100%;
        word-wrap: break-word;
    }}
    .result-card:hover {{
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }}
    .summary {{
        background: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        max-width: 90%;
        width: 400px;
        margin: 30px auto;
        font-size: 18px;
        font-weight: bold;
        color: {"#000000" if dark_mode else "#000000"};
    }}

    /* Responsive adjustments */
    @media (max-width: 600px) {{
        .title {{
            font-size: 2em;
        }}
        .subheader {{
            font-size: 1em;
        }}
        .upload-box {{
            width: 95%;
            padding: 10px;
            font-size: 1em;
        }}
        .result-card {{
            font-size: 0.9em;
            padding: 8px;
        }}
        .summary {{
            width: 95%;
            font-size: 1em;
            padding: 15px;
        }}
        /* Make columns wrap on small screens */
        .stApp > div[data-testid="stHorizontalBlock"] > div {{
            flex-wrap: wrap;
            justify-content: center;
        }}
        /* Force Streamlit columns to stack vertically on mobile */
        .stApp .css-1lcbmhc.e1fqkh3o3 {{
            flex-direction: column !important;
            align-items: center !important;
        }}
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# Maintain files state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'clear_files' not in st.session_state:
    st.session_state.clear_files = False

dark_mode = st.toggle("🌙 Dark Mode", value=False)
set_bg_with_theme("Background_img_2.jpg", "Background_dark_img_2.jpg", dark_mode)

model = load_model("GLD_Binary_Classification_Final.h5")

st.markdown('<div class="title">🌿 GreenGuard 🌿</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">AI-powered Guava Leaf Disease Detector</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-box">📤 Upload Guava Leaf Images (JPG/PNG):</div>', unsafe_allow_html=True)

# Reset file_uploader after clear
key = "file_uploader"
if st.session_state.clear_files:
    key = "file_uploader_new"  # generate new key to force re-render
    st.session_state.clear_files = False

uploaded_files = st.file_uploader(
    "", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True, 
    label_visibility="collapsed",
    key=key
)

# Save uploaded files in session_state
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if st.session_state.uploaded_files:
    analyze_btn = st.button("🔍 Analyze All Uploaded Leaves", key="analyze_btn")
    clear_btn = st.button("❌ Clear Files", key="clear_btn")

    if clear_btn:
        st.session_state.uploaded_files = []
        st.session_state.clear_files = True  # trigger re-render
        st.rerun()

    if analyze_btn:
        healthy_count = 0
        diseased_count = 0

        cols = st.columns(len(st.session_state.uploaded_files))

        for idx, uploaded_file in enumerate(st.session_state.uploaded_files):
            img = Image.open(uploaded_file)
            img_resized = img.resize((256, 256))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            with st.spinner(f'Analyzing {uploaded_file.name}...'):
                time.sleep(1)
                prediction = model.predict(img_array)[0][0]
                if prediction > 0.5:
                    label = "🟢 Healthy Leaf"
                    confidence = prediction
                    healthy_count += 1
                else:
                    label = "🔴 Diseased Leaf"
                    confidence = 1 - prediction
                    diseased_count += 1

            with cols[idx]:
                st.image(img, use_container_width=True, caption=f"File: {uploaded_file.name}")
                st.markdown(f"""
                    <div class='result-card'>
                        Result: {label}<br>
                        Confidence: {confidence*100:.2f}%
                    </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='summary'>
                📊 <u>Summary Dashboard</u><br><br>
                🟢 Healthy Leaves: {healthy_count}<br>
                🔴 Diseased Leaves: {diseased_count}
            </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="upload-box">👈 Please upload one or more guava leaf images to begin analysis.</div>', 
        unsafe_allow_html=True
    )
