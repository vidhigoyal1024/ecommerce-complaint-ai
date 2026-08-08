"""
Streamlit front-end for the AI-based E-Commerce Complaint Review System.

Run:
    streamlit run streamlit_app.py
"""
import os
import sys

import streamlit as st
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import ComplaintClassifier
from src.llm_agent import analyze_with_llm
from src.image_utils import compute_ela

# --------------------------------------------------------------------------
# Page config + styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Complaint Review System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #fafafa; }

    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.6rem;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #ececec;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    .badge-refund      { background: #e6f7ee; color: #0f9d58; }
    .badge-replacement { background: #e8f0fe; color: #1a73e8; }
    .badge-escalation  { background: #fdecea; color: #d93025; }

    .verdict-real { color: #0f9d58; font-weight: 700; }
    .verdict-fake { color: #d93025; font-weight: 700; }

    .pill {
        display: inline-block;
        background: #f1f3f4;
        color: #3c4043;
        border-radius: 8px;
        padding: 0.15rem 0.6rem;
        font-size: 0.85rem;
        margin-right: 0.4rem;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        background-color: #1a1a2e;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #33334d;
        color: white;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached resource loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_text_classifier():
    return ComplaintClassifier()


@st.cache_resource(show_spinner=False)
def load_image_checker():
    from src.image_predict import ImageAuthenticityChecker
    return ImageAuthenticityChecker()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛒 How it works")
    st.markdown(
        """
        1. **Preprocessing** cleans and tokenizes the complaint text.
        2. **CNN + RNN + LSTM** each independently classify the complaint's
           category and severity.
        3. **Model Fusion** blends their outputs into one confident
           prediction.
        4. If a photo is attached, an **Image Authenticity CNN**
           (Error-Level-Analysis based) checks it for signs of digital
           tampering — purely to catch doctored "proof" photos.
        5. The **LLM Agent** retrieves the matching company policy and
           reasons over everything to produce a final recommendation.
        """
    )
    st.divider()
    st.markdown("### ⚙️ Model status")

    try:
        load_text_classifier()
        st.success("Text ensemble (CNN/RNN/LSTM): loaded")
    except FileNotFoundError:
        st.warning("Text ensemble not trained yet.\nRun `python src/train.py`")

    try:
        load_image_checker()
        st.success("Image authenticity CNN: loaded")
    except FileNotFoundError:
        st.info(
            "Image authenticity CNN not trained yet (optional).\n"
            "Run `python scripts/generate_sample_images.py` then "
            "`python src/train_image_cnn.py`"
        )

    st.divider()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_key:
        st.success("LLM Agent: using Claude for reasoning")
    else:
        st.info("LLM Agent: no ANTHROPIC_API_KEY set — using rule-based fallback reasoning")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="app-title">AI-Based E-Commerce Complaint Review</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">CNN + RNN + LSTM ensemble classification, '
    'image-tampering detection, and policy-aware LLM reasoning.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.15], gap="large")

# --------------------------------------------------------------------------
# Left column: input
# --------------------------------------------------------------------------
with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 📝 Submit a complaint")
    complaint_text = st.text_area(
        "Complaint text",
        placeholder="e.g. My package is three weeks late and support won't respond.",
        height=140,
        label_visibility="collapsed",
    )
    uploaded_image = st.file_uploader(
        "Attach a photo (optional — checked for tampering)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Attached photo", use_container_width=True)

    analyze_clicked = st.button("Analyze Complaint", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Try a sample complaint"):
        samples = [
            "My package is three weeks late and support won't respond.",
            "The laptop I bought overheats and shuts down randomly.",
            "I was charged twice for the same order.",
            "I returned the item a month ago and still haven't received my refund.",
        ]
        for s in samples:
            st.markdown(f"- {s}")

# --------------------------------------------------------------------------
# Right column: results
# --------------------------------------------------------------------------
with right:
    if analyze_clicked:
        if not complaint_text.strip():
            st.warning("Please enter a complaint before analyzing.")
        else:
            try:
                clf = load_text_classifier()
            except FileNotFoundError:
                st.error(
                    "The text classification models haven't been trained yet.\n\n"
                    "Run `python src/train.py` in your terminal, then reload this page."
                )
                st.stop()

            with st.spinner("Running CNN / RNN / LSTM ensemble..."):
                prediction = clf.predict(complaint_text)

            image_authenticity = None
            pil_image = None
            if uploaded_image is not None:
                pil_image = Image.open(uploaded_image).convert("RGB")
                try:
                    checker = load_image_checker()
                    with st.spinner("Checking photo authenticity (ELA + CNN)..."):
                        image_authenticity = checker.predict(pil_image)
                except FileNotFoundError:
                    st.info("Image authenticity CNN not trained — skipping photo check.")

            with st.spinner("LLM Agent retrieving policy and reasoning..."):
                llm_result = analyze_with_llm(
                    complaint_text=complaint_text,
                    category=prediction["category"],
                    severity=prediction["severity"],
                    confidence=prediction["category_confidence"],
                    image_authenticity=image_authenticity,
                )

            # ---- Classification card ----
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Classification")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Category", prediction["category"])
                st.progress(prediction["category_confidence"], text=f"Confidence: {prediction['category_confidence']:.0%}")
            with c2:
                st.metric("Severity", prediction["severity"])
                st.progress(prediction["severity_confidence"], text=f"Confidence: {prediction['severity_confidence']:.0%}")
            st.markdown("</div>", unsafe_allow_html=True)

            # ---- Image authenticity card ----
            if pil_image is not None:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown("#### 🖼️ Photo authenticity check")
                if image_authenticity is not None:
                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.image(pil_image, caption="Original", use_container_width=True)
                    with ic2:
                        ela_preview = compute_ela(pil_image)
                        st.image(ela_preview, caption="Error Level Analysis map", use_container_width=True)

                    verdict_class = "verdict-real" if image_authenticity["is_real"] else "verdict-fake"
                    verdict_icon = "✅" if image_authenticity["is_real"] else "🚩"
                    st.markdown(
                        f'{verdict_icon} <span class="{verdict_class}">{image_authenticity["label"]}</span> '
                        f'&nbsp;·&nbsp; confidence {image_authenticity["confidence"]:.0%}',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Real probability: {image_authenticity['real_probability']:.0%}  ·  "
                        f"Fake probability: {image_authenticity['fake_probability']:.0%}"
                    )
                else:
                    st.info("Image authenticity CNN not trained yet — photo was not checked.")
                st.markdown("</div>", unsafe_allow_html=True)

            # ---- Recommendation card ----
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### ✅ Final Recommendation")
            action = llm_result["action"]
            badge_class = {
                "Refund": "badge-refund",
                "Replacement": "badge-replacement",
                "Escalation": "badge-escalation",
            }.get(action, "badge-escalation")
            st.markdown(f'<span class="badge {badge_class}">{action}</span>', unsafe_allow_html=True)
            st.write(llm_result["reasoning"])

            source_label = "Claude LLM Agent" if llm_result["source"] == "llm_agent" else "Rule-based fallback"
            st.markdown(f'<span class="pill">Reasoned by: {source_label}</span>', unsafe_allow_html=True)

            with st.expander("Company policy used"):
                st.write(llm_result["policy_used"])
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 👋 Get started")
        st.write(
            "Enter a complaint on the left (and optionally attach a photo), "
            "then click **Analyze Complaint** to see the ensemble classification, "
            "photo authenticity check, and final recommendation here."
        )
        st.markdown("</div>", unsafe_allow_html=True)
