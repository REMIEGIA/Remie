import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import time

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="CNN Image Classifier Report", layout="wide")

# -------------------------------------------------
# Header (For Report Submission)
# -------------------------------------------------
st.title("🧠 CNN Image Classification using Streamlit")
st.markdown("""
**Student ID:** `G140A001`  
**Model:** Convolutional Neural Network (CNN)  
**Dataset:** CIFAR-10  
**Features:** Training Progress, Accuracy/Loss Curves, Image Prediction, Report Export
""")

st.divider()

# -------------------------------------------------
# 1️⃣ Load Dataset
# -------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# -------------------------------------------------
# 2️⃣ Build CNN Model
# -------------------------------------------------
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------------------------------
# 3️⃣ Training Settings (Sidebar)
# -------------------------------------------------
st.sidebar.header("⚙️ Training Settings")
epochs = st.sidebar.slider("Number of Epochs", 1, 10, 3)
train_button = st.sidebar.button("🚀 Train Model")

# -------------------------------------------------
# 4️⃣ Train Model with Progress & Logging
# -------------------------------------------------
if train_button:
    model = build_model()

    st.subheader("📌 Model Summary")
    with st.expander("Show CNN Architecture"):
        model.summary(print_fn=lambda x: st.text(x))

    progress = st.progress(0)
    status = st.empty()

    history_log = []

    for epoch in range(epochs):
        hist = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=256,
            validation_data=(x_test, y_test),
            verbose=0
        )

        log = {
            "Epoch": epoch + 1,
            "Train Accuracy": hist.history["accuracy"][0],
            "Validation Accuracy": hist.history["val_accuracy"][0],
            "Train Loss": hist.history["loss"][0],
            "Validation Loss": hist.history["val_loss"][0],
        }
        history_log.append(log)

        progress.progress((epoch + 1) / epochs)
        status.text(f"Training Epoch {epoch + 1} / {epochs}")
        time.sleep(0.3)

    st.success("✅ Training Completed")
    st.session_state["model"] = model
    history_df = pd.DataFrame(history_log)

    # -------------------------------------------------
    # 📊 Accuracy & Loss Graphs
    # -------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(history_df["Train Accuracy"], label="Train")
        ax1.plot(history_df["Validation Accuracy"], label="Validation")
        ax1.set_title("📈 Accuracy Curve")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(history_df["Train Loss"], label="Train")
        ax2.plot(history_df["Validation Loss"], label="Validation")
        ax2.set_title("📉 Loss Curve")
        ax2.legend()
        st.pyplot(fig2)

    # -------------------------------------------------
    # 🆕 NEW FEATURE: Download Training Report
    # -------------------------------------------------
    st.subheader("📄 Download Training Report")
    st.download_button(
        label="⬇️ Download CSV Report",
        data=history_df.to_csv(index=False),
        file_name="CNN_Training_Report_G140A001.csv",
        mime="text/csv"
    )

else:
    st.info("Please train the model using the sidebar.")

# -------------------------------------------------
# 5️⃣ Image Upload & Prediction
# -------------------------------------------------
if "model" in st.session_state:
    st.divider()
    st.header("📸 Image Classification")

    uploaded_file = st.file_uploader(
        "Upload an image (JPG/PNG)",
        type=["jpg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).resize((32, 32))
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        prediction = st.session_state["model"].predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.markdown(f"### ✅ Prediction: **{class_names[predicted_class]}**")
            st.markdown(f"Confidence Score: `{confidence:.2f}`")

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(class_names, prediction[0], color="skyblue")
            ax.set_title("Prediction Probability Distribution")
            st.pyplot(fig)

# -------------------------------------------------
# 6️⃣ CSV Upload & Batch Prediction (NEW FEATURE)
# -------------------------------------------------
if "model" in st.session_state:
    st.divider()
    st.header("📂 CSV Batch Image Prediction")

    st.markdown("""
    **CSV Requirements:**
    - Each row represents one image
    - 3072 columns (32×32×3 flattened)
    - Pixel values: 0–255 or 0–1
    """)

    csv_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=["csv"]
    )

    if csv_file:
        try:
            df = pd.read_csv(csv_file)

            st.subheader("📄 Uploaded CSV Preview")
            st.dataframe(df.head())

            if df.shape[1] != 3072:
                st.error("❌ CSV must contain exactly 3072 columns.")
            else:
                # Normalize & reshape
                data = df.values.astype("float32")
                if data.max() > 1:
                    data /= 255.0

                data = data.reshape(-1, 32, 32, 3)

                predictions = st.session_state["model"].predict(data)
                predicted_classes = np.argmax(predictions, axis=1)
                confidence_scores = np.max(predictions, axis=1)

                results_df = pd.DataFrame({
                    "Predicted Class": [class_names[i] for i in predicted_classes],
                    "Confidence Score": confidence_scores
                })

                st.subheader("✅ Prediction Results")
                st.dataframe(results_df)

                # Download results
                st.download_button(
                    label="⬇️ Download Prediction Results (CSV)",
                    data=results_df.to_csv(index=False),
                    file_name="CSV_Prediction_Results_G140A001.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"⚠️ Error processing CSV file: {e}")
# -------------------------------------------------
# 5️⃣ Image Upload, Light Adjustment & Prediction
# -------------------------------------------------
if "model" in st.session_state:
    st.divider()
    st.header("📸 Image Classification with Light Adjustment")

    uploaded_file = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=["jpg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB").resize((32, 32))

        # 🆕 Brightness / Lighting Adjustment
        st.subheader("💡 Adjust Image Brightness")
        brightness = st.slider(
            "Lighting Level",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1
        )

        # Apply brightness
        img_array = np.array(image).astype("float32") / 255.0
        img_array = np.clip(img_array * brightness, 0, 1)

        input_img = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = st.session_state["model"].predict(input_img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(
                (img_array * 255).astype("uint8"),
                caption=f"Adjusted Image (Brightness: {brightness})",
                use_column_width=True
            )

        with col2:
            st.markdown(f"### ✅ Prediction: **{class_names[predicted_class]}**")
            st.markdown(f"Confidence Score: `{confidence:.2f}`")

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(class_names, prediction[0], color="orange")
            ax.set_title("Prediction Probability Distribution")
            st.pyplot(fig)

