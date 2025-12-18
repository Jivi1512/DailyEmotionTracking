import os
import cv2
import time
import requests
import numpy as np
import pandas as pd
from fer import FER
from PIL import Image
import streamlit as st
import tensorflow as tf
import plotly.express as px
from datetime import datetime
import google.generativeai as genai
from scipy.signal import butter, filtfilt
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="EmoTrack AI", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #f8f9fa;
        color: #31333F;
        font-weight: 600;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff4b4b;
        color: white;
        border-color: #ff4b4b;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
    }
    h1 { color: #1e293b; }
    div.block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

SHEET_MAPPING={'users': 'UserDB', 'data': 'DataDB'}

def get_connector():
    return st.connection("gsheets", type=GSheetsConnection)

def load_data(data_type):
    conn=get_connector()
    worksheet_name=SHEET_MAPPING.get(data_type)
    try:
        df=conn.read(worksheet=worksheet_name, ttl=0)
        if data_type == 'users':
            if df.empty: return {}
            df=df.drop_duplicates(subset=['username'])
            df=df.set_index("username")
            return df.to_dict(orient="index")
        elif data_type == 'data':
            if df.empty: return {}
            grouped=df.groupby("username")
            result={}
            for user, group in grouped:
                records=group.drop(columns=["username"]).to_dict(orient="records")
                result[user]=records
            return result
    except Exception:
        return {}

def save_data(data, data_type):
    conn=get_connector()
    worksheet_name=SHEET_MAPPING.get(data_type)
    if data_type == 'users':
        df=pd.DataFrame.from_dict(data, orient='index')
        df.index.name='username'
        df.reset_index(inplace=True)
        conn.update(worksheet=worksheet_name, data=df)
    elif data_type == 'data':
        all_records=[]
        for user, entries in data.items():
            for entry in entries:
                entry['username']=user
                all_records.append(entry)
        df=pd.DataFrame(all_records)
        conn.update(worksheet=worksheet_name, data=df)

USER_DB='users'
DATA_DB='data'

def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def login_page():
    col1, col2, col3=st.columns([1, 1, 1])
    with col2:
        st.title("Login")
        st.markdown("Welcome back! Please enter your credentials.")
        username=st.text_input("Username")
        password=st.text_input("Password", type="password")
        if st.button("Login"):
            users=load_data(USER_DB)
            if username in users and str(users[username]['password']) == password:
                st.session_state['logged_in']=True
                st.session_state['username']=username
                st.session_state['user_info']=users[username]
                st.rerun()
            else:
                st.error("Invalid credentials")

def signup_page():
    st.title("Sign Up")
    col1, col2=st.columns(2)
    with col1:
        new_user=st.text_input("Username")
        new_pass=st.text_input("Password", type="password")
        name=st.text_input("Full Name")
    with col2:
        age=st.number_input("Age", min_value=1)
        occupation=st.text_input("Occupation")
        mood=st.selectbox("Current Mood", ["Happy", "Sad", "Anxious", "Neutral", "Stressed"])
    
    health=st.text_area("Health Conditions")
    
    if st.button("Create Account"):
        users=load_data(USER_DB)
        if new_user in users:
            st.error("User exists")
        else:
            users[new_user]={
                'password': new_pass, 'name': name, 'age': age,
                'occupation': occupation, 'base_mood': mood,
                'health': health, 'joined': str(datetime.now())
            }
            save_data(users, USER_DB)
            st.success("Created! Go to Login.")

def face_detection_page():
    st.title("Real-Time Emotion Detection")
    st.caption("Analyze facial expressions using computer vision.")
    img_file_buffer=st.camera_input("Capture Image")
    
    if img_file_buffer:
        image=Image.open(img_file_buffer)
        img_array=np.array(image)
        detector=FER(mtcnn=True)
        try:
            result=detector.top_emotion(img_array)
            if result:
                emotion, score=result
                col1, col2=st.columns(2)
                with col1:
                    st.image(image, caption="Captured", use_column_width=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Detected Emotion</h3>
                        <h2 style="color: #ff4b4b;">{str(emotion).upper()}</h2>
                        <p>Confidence: {score*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("Save to History"):
                    all_data=load_data(DATA_DB)
                    user_data=all_data.get(st.session_state['username'], [])
                    user_data.append({
                        "date": str(datetime.now()), "type": "face_scan",
                        "emotion": emotion, "score": score, "summary": "N/A"
                    })
                    all_data[st.session_state['username']]=user_data
                    save_data(all_data, DATA_DB)
                    st.success("Entry Saved!")
            else:
                st.warning("No face detected.")
        except Exception as e:
            st.error(f"Error: {e}")

def chatbot_page():
    st.title("AI Therapist")
    api_key=st.secrets.get("GEMINI_API_KEY")
    if not api_key: api_key=st.text_input("API Key", type="password")
    
    if api_key:
        model=init_gemini(api_key)
        if "messages" not in st.session_state: st.session_state.messages=[]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])

        if prompt := st.chat_input("How do you feel?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            user_info=st.session_state['user_info']
            full_prompt=f"Role: Therapist. Client: {user_info['name']}. User: {prompt}"
            
            try:
                response=model.generate_content(full_prompt)
                st.chat_message("assistant").markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
                all_data=load_data(DATA_DB)
                user_data=all_data.get(st.session_state['username'], [])
                user_data.append({
                    "date": str(datetime.now()), "type": "chat_log",
                    "emotion": "N/A", "score": 0, "summary": prompt[:100]
                })
                all_data[st.session_state['username']]=user_data
                save_data(all_data, DATA_DB)
            except Exception as e:
                st.error(f"Error: {e}")

class DummyModel:
    def __init__(self, output_dim=3): self.output_dim=output_dim
    def predict(self, input_data, verbose=0):
        return np.random.dirichlet(np.ones(self.output_dim), size=input_data.shape[0])

class SafeInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs: kwargs['batch_input_shape']=kwargs.pop('batch_shape')
        super().__init__(**kwargs)
    def get_config(self): return super().get_config()

class MockDTypePolicy:
    def __init__(self, name="float32", **kwargs): self._name=name
    def get_config(self): return {"name": self._name}
    @classmethod
    def from_config(cls, config): return cls(**config)

def robust_bandpass_filter(signal, fs=200):
    b, a=butter(4, [4.0/100, 45.0/100], btype='band')
    return filtfilt(b, a, signal, axis=0)

@st.cache_resource
def load_eeg_model():
    model_path="best_eeg_model.keras"
    url="https://raw.githubusercontent.com/Jivi1512/DailyEmotionTracking/main/best_eeg_model.keras"
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
        with st.spinner("Downloading EEG Model..."):
            try:
                r=requests.get(url, stream=True)
                if r.status_code == 200:
                    with open(model_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            except: pass

    try:
        return tf.keras.models.load_model(
            model_path, 
            custom_objects={'InputLayer': SafeInputLayer, 'DTypePolicy': MockDTypePolicy}, 
            compile=False
        ), True
    except Exception as e: 
        st.error(f"REAL ERROR: {e}")  # <--- Add this line to see the error on screen
        return DummyModel(output_dim=7), False

def eeg_page():
    st.title("EEG Analysis")
    st.caption("Multichannel EEG Signal Processing")
    
    model, is_real=load_eeg_model()
    if not is_real: st.warning("Using Simulation Mode")
    
    if st.button("Start EEG Scan"):
        col1, col2=st.columns([2, 1])
        status=st.empty()
        bar=st.progress(0)
        emotions=['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        for i in range(10):
            raw=np.random.normal(0, 1, (800, 62))
            proc=robust_bandpass_filter(raw)
            proc=(proc - np.mean(proc)) / (np.std(proc) + 1e-6)
            input_batch=np.expand_dims(proc, axis=0)
            
            preds=model.predict(input_batch, verbose=0)[0]
            top_idx=np.argmax(preds)
            
            with col1:
                st.line_chart(proc[:100, 0], height=470)
            with col2:
                st.metric("Detected", emotions[top_idx], f"{preds[top_idx]*100:.1f}%")
                st.bar_chart(pd.DataFrame({"Prob": preds}, index=emotions))
            
            status.text(f"Processing batch {i+1}/10...")
            bar.progress((i+1)*10)
            time.sleep(0.5)
        status.success("Complete")

@st.cache_resource
def load_ecg_model():
    model_path="wesad_cnn_lstm_model.keras"
    url="https://raw.githubusercontent.com/Jivi1512/DailyEmotionTracking/main/wesad_cnn_lstm_model.keras"
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
        with st.spinner("Downloading WESAD ECG Model..."):
            try:
                r=requests.get(url, stream=True)
                if r.status_code == 200:
                    with open(model_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            except: pass

    try:
        return tf.keras.models.load_model(
            model_path, custom_objects={'InputLayer': SafeInputLayer, 'DTypePolicy': MockDTypePolicy}, compile=False
        ), True
    except: 
        return DummyModel(output_dim=3), False

def ecg_page():
    st.title("ECG Emotion Monitor")
    st.caption("Detects Baseline, Amusement, or Neutral states using CNN-LSTM on ECG signals.")
    
    model, is_real=load_ecg_model()
    
    if is_real:
        st.success("WESAD CNN-LSTM Model Loaded")
    else:
        st.warning("Simulation Mode (Real model failed to load)")

    if st.button("Start Live ECG Analysis"):
        col1, col2=st.columns([3, 1])
        status=st.empty()
        bar=st.progress(0)
        
        labels=['Baseline', 'Amusement', 'Neutral']
        
        for i in range(20):
            t=np.linspace(0, 10, 1000)
            ecg_wave=np.sin(t*3) + 0.5 * np.sin(t*12) + np.random.normal(0, 0.2, 1000)
            ecg_proc=(ecg_wave - np.mean(ecg_wave)) / np.std(ecg_wave)
            input_batch=ecg_proc.reshape(1, 1000, 1)
            
            preds=model.predict(input_batch, verbose=0)[0]
            top_idx=np.argmax(preds)
            confidence=preds[top_idx]
            
            with col1:
                st.line_chart(ecg_wave[:200], height=250)
            with col2:
                st.metric("State", labels[top_idx], f"{confidence*100:.1f}%")
                st.bar_chart(pd.DataFrame({"Prob": preds}, index=labels))
                
            status.text(f"Analyzing Segment {i+1}/20...")
            bar.progress((i+1)*5)
            time.sleep(0.3)
            
        status.success("ECG Session Complete")

def dashboard_page():
    user=st.session_state['user_info']
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; color: white; margin-bottom: 20px;">
        <h1>Hi, {user['name']}!</h1>
        <p style="font-size: 1.2em; opacity: 0.9;">Ready to track your emotional well-being today?</p>
        <div style="margin-top: 15px; display: flex; gap: 20px;">
            <span>Occupation: {user.get('occupation', 'N/A')}</span>
            <span>Base Mood: {user.get('base_mood', 'Neutral')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    all_data=load_data(DATA_DB)
    history=all_data.get(st.session_state['username'], [])
    
    df=pd.DataFrame(history)
    
    dom_emo="N/A"
    if not df.empty and 'emotion' in df.columns:
        valid=df[df['emotion'] != "N/A"]['emotion'].dropna()
        if not valid.empty:
            mode_result=valid.mode()
            if not mode_result.empty:
                dom_emo=mode_result.iloc[0]

    total=len(history)
    last=history[-1]['date'][:10] if history else "N/A"

    col1, col2, col3=st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>Total Logs</h3><h2>{total}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>Last Activity</h3><h2>{last}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>Dominant Mood</h3><h2>{str(dom_emo).title()}</h2></div>", unsafe_allow_html=True)

    st.divider()

    if not df.empty and 'score' in df.columns:
        c1, c2=st.columns([2, 1])
        with c1:
            st.subheader("Emotion Intensity Trend")
            fig=px.line(df, x='date', y='score', color='emotion', markers=True, template="plotly_white")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Mood Distribution")
            fig2=px.pie(df, names='emotion', hole=0.5, template="plotly_white")
            fig2.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

def main():
    if 'logged_in' not in st.session_state: st.session_state['logged_in']=False

    if not st.session_state['logged_in']:
        menu=st.sidebar.radio("Navigation", ["Login", "Sign Up"])
        if menu == "Login": login_page()
        else: signup_page()
    else:
        with st.container():
            col_logo, col_nav, col_logout=st.columns([1, 6, 1])
            with col_logo:
                st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50)
            
            with col_nav:
                if "page" not in st.session_state: st.session_state.page="Dashboard"
                b1, b2, b3, b4, b5=st.columns(5)
                if b1.button("Dashboard"): st.session_state.page="Dashboard"
                if b2.button("AI Chat"): st.session_state.page="Chat"
                if b3.button("Face"): st.session_state.page="Face"
                if b4.button("EEG"): st.session_state.page="EEG"
                if b5.button("ECG"): st.session_state.page="ECG"
            
            with col_logout:
                if st.button("Logout"):
                    st.session_state['logged_in']=False
                    st.rerun()
            st.divider()

        page=st.session_state.page
        if page == "Dashboard": dashboard_page()
        elif page == "Chat": chatbot_page()
        elif page == "Face": face_detection_page()
        elif page == "EEG": eeg_page()
        elif page == "ECG": ecg_page()

if __name__ == "__main__":
    main()


