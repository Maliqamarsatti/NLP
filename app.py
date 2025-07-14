# app.py

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 1) Try to import our RAG assistant, catching any errors
try:
    from rag_chatbot.chatbot import init_chatbot, ask_question
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    INIT_ERROR = str(e)

st.set_page_config(page_title="Coffee Leaf Disease Detector & Assistant", layout="centered")
st.title("☕ Coffee Leaf Disease Detection & RAG Assistant 🌿")

# 2) Show any import/init errors up front
if not RAG_AVAILABLE:
    st.error("🔴 Could not load the RAG assistant:\n\n" + INIT_ERROR)
    st.stop()

# 3) Load YOLOv8 model (make sure you have yolov8n.pt in this folder)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 4) Initialize the chatbot (this will block until done)
with st.spinner("Loading the RAG assistant (this may take ~30s)…"):
    qa_chain = init_chatbot()
st.success("✅ RAG assistant ready!")

# 5) Image upload + detection
st.header("1️⃣ Upload a Coffee Leaf Image")
uploaded_file = st.file_uploader("Choose an image…", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded leaf", use_container_width=True)

    with st.spinner("Detecting diseases…"):
        results = model(img)
        names   = results[0].names
        dets    = [names[int(c)] for c in results[0].boxes.cls]

    if dets:
        diseases = sorted(set(dets))
        st.success(f"Detected disease(s): {', '.join(diseases)}")
        # Auto‐generate a remedy
        prompt = f"What is {', '.join(diseases)} on coffee leaves and how should it be treated?"
        with st.spinner("Generating remedy…"):
            remedy = ask_question(prompt, qa_chain)
        st.subheader("💊 Suggested Remedy")
        st.write(remedy)

        # Follow‐up Q&A
        st.markdown("---")
        st.header("2️⃣ Ask a Follow-up Question")
        q = st.text_input("Your question about treatment, prevention, etc.")
        if q:
            with st.spinner("Thinking…"):
                ans = ask_question(q, qa_chain)
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {ans}")
    else:
        st.info("No disease detected — leaf looks healthy! 🎉")

# 6) Sidebar instructions
with st.sidebar:
    st.markdown("## 📋 How to Use")
    st.markdown(
        "- Upload a coffee leaf image\n"
        "- View detected diseases\n"
        "- See automated remedy suggestions\n"
        "- Ask follow-up questions for more detail\n"
    )
