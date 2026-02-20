
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import plotly.express as px
import pandas as pd

# --- App Design & Visuals ---
st.set_page_config(page_title="Ensemble Image Classifier", page_icon="✨", layout="wide")

# Custom CSS for Modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    /* Main Background - Soft Gradient */
    .stApp {
        background: radial-gradient(circle at top left, #f8fafc, #e2e8f0);
    }

    /* Header/Hero Section */
    .hero-container {
        text-align: center;
        padding: 3rem 2rem;
        background: white; /* Clean white background */
        border-radius: 20px; /* Fully rounded corners as requested */
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.5);
        width: 100%; /* Ensure full width alignment */
        box-sizing: border-box;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .stCard {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        width: 100%; /* Ensure full width alignment */
        box-sizing: border-box;
    }
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.12);
    }

    /* Custom Button Style */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }

    /* File Uploader area */
    [data-testid='stFileUploader'] {
        width: 100%;
    }
    [data-testid='stFileUploader'] section {
        padding: 8rem; /* Dramatically increased padding ~+150px height */
        background-color: white;
        border: 3px dashed #cbd5e1;
        border-radius: 20px; /* Rounded corners */
        text-align: center;
        transition: border-color 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    [data-testid='stFileUploader'] section:hover {
        border-color: #3b82f6;
        background-color: #f8fafc;
    }
    [data-testid='stFileUploader'] .uploadedFileName {
        color: #3b82f6;
        font-weight: 600;
    }

    /* Remove extra internal spacing from uploader containers */
    [data-testid='stFileUploader'] > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    .st-emotion-cache-1l95nvm, .e16n7gab9 {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 5rem;
        padding: 2rem;
        color: #94a3b8;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Models ---
@st.cache_resource
def load_models():
    eff = tf.keras.models.load_model(os.path.join("bestmodel", "efficientnet_base.keras"), compile=False)
    mob = tf.keras.models.load_model(os.path.join("bestmodel", "mobilenet_base.keras"), compile=False)
    res = tf.keras.models.load_model(os.path.join("bestmodel", "resnet_tuned.keras"), compile=False)
    return eff, mob, res

try:
    model_eff, model_mob, model_res = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure the 'bestmodel' directory exists.")
    st.stop()

# Ensure these align with the model's training indices (usually alphabetical)
class_names = ['Bicycle', 'Bus', 'Car', 'Door', 'Motorcycle', 'Person', 'Stairs', 'Traffic Light']

# --- Preprocessing ---
def preprocess_image(img):
    img = img.resize((224, 224))
    arr = np.array(img)
    if arr.shape[-1] == 4:  # RGBA to RGB
        arr = arr[..., :3]
    arr = arr[np.newaxis, ...]
    return arr

# --- Ensemble Prediction ---
def predict_ensemble(img):
    arr = preprocess_image(img)
    pred_eff = model_eff.predict(arr)
    arr_scaled = (arr / 127.5) - 1
    pred_mob = model_mob.predict(arr_scaled)
    pred_res = model_res.predict(arr_scaled)
    avg_pred = np.mean([pred_eff, pred_mob, pred_res], axis=0)
    return class_names[np.argmax(avg_pred)], avg_pred[0]


# --- Main UI Modern Layout ---
# Use a consistent column ratio for the whole flow to fix alignment
# Use a consistent column ratio for the whole flow to fix alignment
# [1, 3, 1] makes it wider (60% width vs 50%)
c1, c2, c3 = st.columns([1, 3, 1])

with c2:
    # --- Hero Section ---
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Outdoor Vision AI</h1>
        <p class="hero-subtitle">Object Recognition powered by Multi-Model Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Instructions / Walkthrough ---
    with st.expander("How to use / Walkthrough"):
        st.markdown("""
        1.  **Upload**: Click the 'Browse files' button to select outdoor image (JPEG/PNG).
        2.  **Analyze**: The ensemble of AI models (EfficientNet, MobileNet, ResNet) will process the image.
        3.  **Result**: View the predicted class and the confidence score.
        4.  **Details**: Explore the interactive charts and data table for deeper insights.
        """)
        st.info("💡 Detectable classes: Bicycle, Bus, Car, Door, Motorcycle, Person, Stairs, Traffic Light")

    st.write("") # Spacer
    uploaded_file = st.file_uploader("📂 Upload an image (JPG, PNG) • Max 200MB", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        import base64
        import io

        # Convert image to Base64 for CSS injection
        img = Image.open(uploaded_file).convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=70) 
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Dynamic CSS: Refined to specifically target the file icon and NOT the delete button
        st.markdown(f"""
        <style>
            /* Shrink the main dropzone height since file is uploaded */
            [data-testid='stFileUploader'] section {{
                padding: 1.5rem !important;
            }}
            /* Target ONLY the file icon SVG (first child) and hide it */
            [data-testid='stFileUploaderFile'] div[class*="e16n7gab14"] svg {{
                display: none !important;
            }}
            /* Injected Thumbnail */
            [data-testid='stFileUploaderFile'] div[class*="e16n7gab14"] {{
                background-image: url('data:image/jpeg;base64,{img_base64}');
                background-size: cover;
                background-position: center;
                width: 60px !important;
                height: 60px !important;
                border-radius: 8px;
                margin-right: 15px;
                border: 1px solid #e2e8f0;
                display: block !important;
            }}
            /* Ensure the info bar looks clean and fills the width */
            [data-testid='stFileUploaderFile'] {{
                display: flex !important;
                align-items: center !important;
                width: 100% !important; /* Force full width alignment */
                box-sizing: border-box !important;
                padding: 0.8rem 1rem !important;
                background: white !important;
                border-radius: 12px !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
                margin-top: 10px;
            }}
        </style>
        """, unsafe_allow_html=True)

        st.write("") # Spacer
        
        # Removed the extra st.image call here to fix the "2 images" issue
        
        # Prediction Logic
        with st.spinner("🤖 AI Models are analyzing the scene..."):
            label, probs = predict_ensemble(img)
            
        # Display Result Card
        st.markdown(f"""
        <div class="stCard">
            <h3 style="text-align: center; color: #64748b; margin-bottom: 0.5rem; font-weight: 500;">PREDICTION RESULT</h3>
            <h1 style="text-align: center; color: #3b82f6; font-size: 3.5rem; margin: 0; background: linear-gradient(90deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{label.upper()}</h1>
            <p style="text-align: center; color: #64748b; margin-top: 0.5rem;">Confidence Score: <b>{float(np.max(probs))*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Interactive Analysis")
        
        # Prepare Data for Plotly
        df_probs = pd.DataFrame({
            'Class': class_names,
            'Probability': probs
        })
        df_probs['Probability %'] = df_probs['Probability'] * 100
        df_probs['Label'] = df_probs['Class']
        
        # Sort for better visualization
        df_probs = df_probs.sort_values(by='Probability', ascending=True)

        # Plotly Bar Chart
        fig = px.bar(
            df_probs, 
            x='Probability', 
            y='Class', 
            orientation='h',
            text='Probability %',
            color='Probability',
            color_continuous_scale='Bluyl', # Soft blue-yellow-purple scale
            title='Model Confidence per Class'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Probability (0-1)",
            yaxis_title=None,
            showlegend=False,
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data Table Expander
        with st.expander("🔎 View Raw Data Table"):
            st.dataframe(
                df_probs.sort_values(by='Probability', ascending=False).style.format({'Probability': '{:.4f}', 'Probability %': '{:.2f}%'}),
                use_container_width=True
            )

# Footer
st.markdown("""
<div class="footer">
    <p>© 2026 Outdoor Vision AI. All rights reserved. | Powered by TensorFlow & Streamlit</p>
    <p style="font-size: 0.75rem;">Privacy Policy • Terms of Service</p>
</div>
""", unsafe_allow_html=True)