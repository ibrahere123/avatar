import streamlit as st
import numpy as np
import re, tempfile, os, base64, json
import sounddevice as sd
from scipy.io.wavfile import write
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from gtts import gTTS

try:
    import faiss
    faiss_import_error = None
except Exception as e:
    faiss = None
    faiss_import_error = e

import speech_recognition as sr

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="AI Avatar Conversational (Bahasa)", page_icon="🗣️", layout="wide")
api_key = "AIzaSyAtUVXherGjYaT0ZTVmHpjxsC5jQHbNN4s"
genai.configure(api_key=api_key)

# ---------------------------
# Helpers
# ---------------------------
def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return (audio / max_val * 32767).astype(np.int16)

def load_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def extract_qa_pairs(text: str):
    blocks = re.split(r"\n(?=Apa|Apakah|Bagaimana|Kapan|Di mana|Siapa)", text)
    qa_pairs = []
    for block in blocks:
        parts = block.strip().split("\n", 1)
        if len(parts) == 2:
            q, a = parts
            qa_pairs.append((q.strip(), a.strip()))
    return qa_pairs

# ---------------------------
# TTS + Lottie Control
# ---------------------------
def play_tts_with_lottie_control(key, text, lottie_json, lang="id"):
    # Generate TTS audio
    tts = gTTS(text=text, lang=lang)
    filename = f"tts_{key}.mp3"
    tts.save(filename)
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    os.remove(filename)
    b64_audio = base64.b64encode(audio_bytes).decode()

    # Convert Lottie JSON to base64
    lottie_data = json.dumps(lottie_json)
    b64_lottie = base64.b64encode(lottie_data.encode()).decode()

    # HTML + JS for synchronized playback
    html_code = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <div style="display:flex; flex-direction:column; align-items:center;">
        <lottie-player 
            id="lottieAnim" 
            autoplay 
            loop 
            mode="normal" 
            style="width:300px;height:400px;" 
            src='data:application/json;base64,{b64_lottie}'>
        </lottie-player>
        <audio id="ttsAudio" autoplay>
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
        </audio>
    </div>
    <script>
        const audio = document.getElementById("ttsAudio");
        const anim = document.getElementById("lottieAnim");

        // Pause Lottie initially
        anim.pause();

        audio.onplay = () => {{ anim.play(); }};
        audio.onpause = () => {{ anim.pause(); }};
        audio.onended = () => {{ anim.pause(); }};
    </script>
    """
    st.components.v1.html(html_code, height=500)

# ---------------------------
# STT
# ---------------------------
def record_and_transcribe(duration=10, fs=16000, device_index=None, language="id-ID"):
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=device_index)
        sd.wait()
        audio = normalize_audio(audio)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
            tmp_wav = tmpf.name
        write(tmp_wav, fs, audio)
        st.audio(tmp_wav, format='audio/wav')
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_wav) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=language)
        os.remove(tmp_wav)
        return text
    except Exception as e:
        try:
            mic = sr.Microphone()
            recognizer = sr.Recognizer()
            st.info("🎤 Menggunakan fallback mikrofon...")
            with mic as source:
                audio_data = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            text = recognizer.recognize_google(audio_data, language=language)
            return text
        except Exception as e2:
            st.error(f"STT Error (fallback): {e2}")
            return None

# ---------------------------
# State
# ---------------------------
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Preload PDF FAQ
# ---------------------------
st.title("🗣️ Conversational AI Avatar — Bahasa Indonesia")
st.caption("Tanya jawab dengan suara atau teks (FAQ → FAISS → Gemini → TTS).")

faq_file_path = r"D:\indonasianavatar\static\QA_GO_GYM.pdf"
threshold = 0.35
top_k = 1

if os.path.exists(faq_file_path):
    raw_text = load_pdf(faq_file_path)
    kb_text = normalize_whitespace(raw_text)
    if kb_text:
        if faiss is None:
            st.error("⚠️ FAISS tidak tersedia. Instalasi: pip install faiss-cpu")
        else:
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
            qa_pairs = extract_qa_pairs(kb_text)
            if qa_pairs:
                questions = [q for q,_ in qa_pairs]
                embeddings = model.encode(questions, convert_to_numpy=True)
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                st.session_state.qa_pairs = qa_pairs
                st.session_state.index = index
                st.session_state.embeddings = embeddings
                st.session_state.kb_ready = True
                st.success(f"{len(qa_pairs)} Q–A dimuat dari PDF '{faq_file_path}'.")
    else:
        st.error(f"⚠️ PDF FAQ tidak ditemukan di path: {faq_file_path}")

# ---------------------------
# Retrieval + Gemini
# ---------------------------
def retrieve(query, k):
    if faiss is None or not st.session_state.kb_ready:
        return []
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    q_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D,I = st.session_state.index.search(q_vec,k)
    return [(st.session_state.qa_pairs[idx],float(score)) for idx,score in zip(I[0],D[0])]

def generate_with_gemini(context_snippets, history, user_input):
    context_text = "\n".join([f"Q: {q}\nA: {a}" for (q, a), _ in context_snippets])
    past = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])
    prompt = f""" Anda adalah asisten AI yang ramah dan profesional yang menjawab dalam Bahasa Indonesia sehari-hari.
📌 Aturan penting:
1. Gunakan HANYA informasi dari konteks FAQ yang diberikan.
2. Jika jawaban tidak ada dalam konteks, katakan dengan sopan: "Maaf, saya tidak tahu jawabannya berdasarkan informasi yang ada."
3. Jawablah dengan singkat, jelas, dan terdengar alami.
4. Jangan mengarang fakta di luar dokumen.
5. Gunakan gaya percakapan agar terasa seperti dialog manusia.
Konteks FAQ: {context_text}
Riwayat percakapan: {past}
Pertanyaan pengguna: {user_input}
Jawaban AI (dalam Bahasa Indonesia yang natural): """
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text

# ---------------------------
# Load Lottie JSON
# ---------------------------
lottie_animation = None
lottie_path = r"D:\indonasianavatar\static\ava.json"
if os.path.exists(lottie_path):
    with open(lottie_path, "r", encoding="utf-8") as f:
        lottie_animation = json.load(f)

# ---------------------------
# Voice Input Section
# ---------------------------
st.subheader("🎙 Bicara")
if st.button("Mulai Rekam 5 detik"):
    st.info("🎤 Silakan mulai berbicara sekarang...")
    user_msg = record_and_transcribe(duration=10)
    if user_msg:
        st.success(f"Anda mengatakan: {user_msg}")
        st.session_state.history.append({"role":"user","content":user_msg})
        with st.chat_message("assistant"):
            if st.session_state.kb_ready:
                results = retrieve(user_msg, top_k)
                best_score = results[0][1]
                if best_score < threshold:
                    reply = "Maaf, saya tidak memiliki informasi itu dalam dokumen saat ini."
                else:
                    reply = generate_with_gemini(results, st.session_state.history, user_msg)
            else:
                reply = "PDF FAQ belum dimuat."
            st.write(reply)
            if lottie_animation:
                play_tts_with_lottie_control('reply', reply, lottie_animation)
            st.session_state.history.append({"role":"assistant","content":reply})

# ---------------------------
# Text Input Section
# ---------------------------
st.subheader("⌨️ Tulis Pertanyaan")
user_text = st.chat_input("Ketik pertanyaan Anda di sini...")
if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})
    with st.chat_message("assistant"):
        if st.session_state.kb_ready:
            results = retrieve(user_text, top_k)
            best_score = results[0][1]
            if best_score < threshold:
                reply = "Maaf, saya tidak memiliki informasi itu dalam dokumen saat ini."
            else:
                reply = generate_with_gemini(results, st.session_state.history, user_text)
        else:
            reply = "PDF FAQ belum dimuat."
        st.write(reply)
        if lottie_animation:
            play_tts_with_lottie_control('reply', reply, lottie_animation)
        st.session_state.history.append({"role":"assistant","content":reply})

# ---------------------------
# Render history
# ---------------------------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

st.info("🎙 Bicara dengan mikrofon atau ⌨️ ketik teks → AI jawab dengan teks + suara Bahasa Indonesia.")
