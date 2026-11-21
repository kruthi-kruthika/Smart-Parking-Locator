
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🚗",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #0B1120;
    color: #E2E8F0;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #F8FAFC !important;
}

p, label, li {
    color: #CBD5E1 !important;
}

[data-testid="stSidebar"] {
    background-color: #111B2D;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

[data-testid="stMetric"] {
    background: #172337;
    border: 1px solid #33445F;
    padding: 18px;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
}

[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
}

[data-testid="stFileUploader"] {
    background: #172337;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #33445F;
}

[data-testid="stFileUploader"] section {
    background: #1E2D43;
    border: 2px dashed #38BDF8;
    border-radius: 12px;
}

[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #F1F5F9 !important;
}

[data-testid="stFileUploader"] button {
    background-color: #38BDF8 !important;
    color: #0B1120 !important;
    border: none !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] button p {
    color: #0B1120 !important;
}

div.stButton > button {
    background: #38BDF8;
    color: #0B1120;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: #7DD3FC;
    color: #0B1120;
}

[data-testid="stAlert"] {
    background-color: #172337 !important;
    border: 1px solid #33445F !important;
    border-radius: 12px;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #FFFFFF !important;
}

hr {
    border-color: #33445F !important;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.markdown("""
    <div style="font-size:25px;font-weight:800;color:#F8FAFC;">
        🚗 Smart Parking AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Model Information")

    st.write("**AI Model:** CNN")
    st.write("**Framework:** PyTorch")
    st.write("**Classes:** Vacant / Occupied")
    st.write("**Input Size:** 96 × 96")

    st.markdown("---")

    st.markdown("### Project")

    st.write("Smart Parking Locator")

    st.caption(
        "AI-based parking slot classification "
        "using deep learning."
    )


# ---------------- MODEL ----------------
class ParkingCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():

    model = ParkingCNN()

    checkpoint = torch.load(
        "models/parking_cnn.pth",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

# ---------------- HEADER ----------------
st.markdown(
    """
    <div style="background:linear-gradient(120deg,#172337,#1C2F49);
                padding:30px;
                border-radius:20px;
                border:1px solid #33445F;
                margin-bottom:25px;">
        <div style="color:#38BDF8;font-size:13px;font-weight:700;letter-spacing:2px;">
            AI PARKING INTELLIGENCE
        </div>
        <div style="font-size:36px;font-weight:800;color:white;margin-top:10px;">
            🚗 Smart Parking Locator
        </div>
        <div style="color:#CBD5E1;font-size:16px;margin-top:10px;">
            Intelligent parking-space classification powered by deep learning.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- MODEL STATUS ----------------
st.markdown(
    """
    <div style="background:#123A2A;
                border:1px solid #22C55E;
                padding:14px;
                border-radius:12px;
                color:white;
                font-weight:700;">
        🟢 AI MODEL READY
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------- MODEL OVERVIEW ----------------
st.markdown("## MODEL OVERVIEW")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("AI Architecture", "CNN")

with c2:
    st.metric("Test Accuracy", "97.8%")

with c3:
    st.metric("Classification Classes", "2")

st.markdown(
    "<p style='color:#CBD5E1; font-size:15px;'>"
    "Upload a parking slot image to check whether it is vacant or occupied."
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# ---------------- IMAGE ANALYSIS ----------------
st.markdown("## 📷 Parking Slot Analysis")

st.write(
    "Upload an image of a **single parking slot** "
    "to analyze it using the trained AI model."
)
st.markdown("### 📤 Upload Parking Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)


# ---------------- TRANSFORMATION ----------------
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])


# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    model = load_model()

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = F.softmax(output, dim=1)[0]

    vacant_probability = probabilities[0].item() * 100
    occupied_probability = probabilities[1].item() * 100

    predicted_class = torch.argmax(probabilities).item()

    col1, col2 = st.columns([1, 1])

    # -------- IMAGE --------
    with col1:

        st.image(
            image,
            caption="Uploaded Parking Image",
            use_container_width=True
        )

    # -------- AI RESULT --------
    with col2:

        st.markdown("### 🧠 AI Prediction")

        if predicted_class == 0:

            st.markdown("""
            <div style="
                background:#123A2A;
                border:1px solid #22C55E;
                padding:18px;
                border-radius:12px;
                color:#FFFFFF;
                font-size:22px;
                font-weight:bold;
            ">
                🟢 VACANT PARKING SLOT
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="
                background:#3A1720;
                border:1px solid #EF4444;
                padding:18px;
                border-radius:12px;
                color:#FFFFFF;
                font-size:22px;
                font-weight:bold;
            ">
                🔴 OCCUPIED PARKING SLOT
            </div>
            """, unsafe_allow_html=True)


        # Confidence
        confidence = probabilities[predicted_class].item() * 100

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


        # Probability Distribution
        st.markdown("### Probability Distribution")

        st.write(f"🟢 Vacant: **{vacant_probability:.2f}%**")

        st.progress(float(probabilities[0]))

        st.write(f"🔴 Occupied: **{occupied_probability:.2f}%**")

        st.progress(float(probabilities[1]))


    # ---------------- AI ANALYSIS ----------------
    st.markdown("---")

    st.markdown("## AI Analysis")

    if predicted_class == 0:

        st.success(
            "The AI model classified this image "
            "as a VACANT parking slot."
        )

    else:

        st.error(
            "The AI model classified this image "
            "as an OCCUPIED parking slot."
        )


else:

    st.info("Upload a parking-slot image to start AI analysis.")


# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    color:#94A3B8;
    padding:10px;
">
    Smart Parking Locator | AI-Powered Parking Classification
</div>
""", unsafe_allow_html=True)
