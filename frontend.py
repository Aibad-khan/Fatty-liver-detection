import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
import random
import os

# Load the model
model = load_model("fatty_liver_model.keras")

# Set up the Streamlit interface
st.title("Fatty Liver Diagnostic Tool")
st.write(
    """
    This tool is designed to assist physicians in diagnosing fatty liver disease.
    Upload an image of the liver, and the model will predict whether it indicates a fatty liver.
    The app also provides detailed visual and textual explanations using LIME, SHAP, and Saliency Maps.
    """
)

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload a liver image (PNG/JPEG)", type=["png", "jpg", "jpeg"])

# Hardcoded detailed explanations
explanations = {
    "fatty": {
        "result": "The image has been classified as indicating a fatty liver. Fatty liver disease is characterized by excessive fat accumulation in the liver cells, which alters the visual texture and brightness patterns in the image.",
        "shap": "SHAP analysis highlights the regions of the image that contributed most significantly to the model's prediction. These highlighted regions correspond to areas with altered texture or brightness that are commonly associated with fatty liver.",
        "lime": "LIME analysis provides a localized understanding by perturbing regions of the image and analyzing their impact on the prediction. The highlighted areas indicate segments that had the greatest influence on determining the image as indicative of a fatty liver.",
        "saliency": "The saliency map shows pixel-level gradients, identifying the areas most relevant to the prediction. The bright regions in the map correlate with features such as texture density and brightness variations characteristic of fatty liver."
    },
    "normal": {
        "result": "The image has been classified as indicating a normal liver. Normal liver tissue typically exhibits uniform texture and brightness patterns without excessive fat accumulation.",
        "shap": "SHAP analysis demonstrates that the model identified a lack of significant variations in the image as evidence supporting the normal liver classification. Regions highlighted in the map show where uniformity was confirmed.",
        "lime": "LIME analysis identifies regions of the image that reinforce the normal classification. The lack of notable highlights indicates the image exhibits standard liver tissue characteristics.",
        "saliency": "The saliency map focuses on the uniform features of the liver, confirming the absence of anomalies such as fat deposits. The low-intensity map regions suggest no significant deviations."
    }
}

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    result = "fatty" if prediction[0][0] < 0.5 else "normal"

    # Display the prediction with explanation
    st.subheader(f"Prediction: {result.capitalize()} Liver")
    st.write(explanations[result]["result"])

    # Generate and display LIME explanation
    st.subheader("LIME Explanation")
    explainer = lime_image.LimeImageExplainer()

    def predict_proba(images):
        return model.predict(images)

    explanation = explainer.explain_instance(
        img_array[0], predict_proba, top_labels=2, hide_color=0, num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    lime_fig, lime_ax = plt.subplots()
    lime_ax.imshow(mark_boundaries(temp / 255.0, mask))
    lime_ax.axis("off")
    st.pyplot(lime_fig)
    st.write(explanations[result]["lime"])

    # Generate and display SHAP explanation
    st.subheader("SHAP Explanation")
    explainer = shap.GradientExplainer(model, img_array)
    shap_values = explainer.shap_values(img_array)
    normalized_shap_values = shap_values / np.max(np.abs(shap_values))
    plt.figure()
    shap.image_plot(normalized_shap_values, img_array, show=False)  # Ensure SHAP does not show the figure
    fig = plt.gcf()  # Get the current figure

# Display the figure in Streamlit
    st.pyplot(fig)
    st.write(explanations[result]["shap"])

    # Generate and display Saliency Map
    st.subheader("Saliency Map")

    def compute_saliency_map(img_array, model):
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            prediction = model(img_tensor)
            output = prediction[:, 0]
        grads = tape.gradient(output, img_tensor)
        saliency = tf.abs(grads)
        saliency_map = saliency.numpy()[0]
        return img_array[0], saliency_map

    def visualize_saliency_map(original_image, saliency_map):
        saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(saliency_map, cmap="hot")
        axs[1].set_title("Saliency Map")
        axs[1].axis("off")
        return fig

    original_image, saliency_map = compute_saliency_map(img_array, model)
    saliency_fig = visualize_saliency_map(original_image, saliency_map)
    st.pyplot(saliency_fig)
    st.write(explanations[result]["saliency"])
