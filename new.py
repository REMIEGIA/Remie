import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

st.set_page_config(page_title="CNN Image Classifier Report", layout="wide")
st.title("🧠 CNN Image Classification (Student ID: G140A001)")

# -------------------------------------------------
# Load Pre-trained Model
# -------------------------------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("cnn_model_G140A001.h5")

model = load_cnn_model()

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# -------------------------------------------------
# 1️⃣ Image Prediction
# -------------------------------------------------
st.header("📸 Single Image Prediction")
uploaded_image = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "png"], key="single_pred")
if uploaded_image:
    image = Image.open(uploaded_image).resize((32,32))
    img_array = np.expand_dims(np.array(image)/255.0, axis=0)

    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.image(image, caption="Uploaded Image")
    st.markdown(f"**Prediction:** {class_names[pred_class]}")
    st.markdown(f"**Confidence:** {confidence:.2f}")

    # Show probability distribution
    fig, ax = plt.subplots(figsize=(5,3))
    ax.barh(class_names, prediction[0], color="skyblue")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)

# -------------------------------------------------
# 2️⃣ CSV Batch Prediction
# -------------------------------------------------
st.header("📂 CSV Batch Prediction")
csv_file = st.file_uploader("Upload CSV file (3072 columns)", type=["csv"], key="csv_pred")
if csv_file:
    df = pd.read_csv(csv_file)
    if df.shape[1] != 3072:
        st.error("CSV must have exactly 3072 columns.")
    else:
        data = df.values.astype("float32")
        if data.max() > 1:
            data /= 255.0
        data = data.reshape(-1,32,32,3)

        predictions = model.predict(data)
        pred_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)

        results_df = pd.DataFrame({
            "Predicted Class": [class_names[i] for i in pred_classes],
            "Confidence Score": confidence_scores
        })

        st.dataframe(results_df)
        st.download_button(
            "Download Predictions CSV",
            results_df.to_csv(index=False),
            file_name="CSV_Prediction_Results_G140A001.csv",
            mime="text/csv"
        )

# -------------------------------------------------
# 3️⃣ Image Brightness Adjustment
# -------------------------------------------------
st.header("💡 Image Brightness Adjustment")
uploaded_bright = st.file_uploader("Upload image (JPG/PNG) for brightness", type=["jpg","png"], key="bright_pred")
if uploaded_bright:
    image = Image.open(uploaded_bright).convert("RGB").resize((32,32))
    brightness = st.slider("Adjust Brightness", 0.5, 2.0, 1.0, 0.1)

    img_array = np.array(image)/255.0
    img_array = np.clip(img_array*brightness, 0,1)
    input_img = np.expand_dims(img_array, axis=0)

    prediction = model.predict(input_img)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.image((img_array*255).astype("uint8"), caption=f"Adjusted Image (Brightness {brightness})")
    st.markdown(f"**Prediction:** {class_names[pred_class]}")
    st.markdown(f"**Confidence:** {confidence:.2f}")
