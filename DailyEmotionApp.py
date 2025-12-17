import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from fer import FER
import google.generativeai as genai
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import tensorflow as tf
from scipy.signal import butter, filtfilt
import time

st.set_page_config(page_title="EmoTrack AI", layout="wide")

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
    st.title("Login")
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
    new_user=st.text_input("Username")
    new_pass=st.text_input("Password", type="password")
    st.markdown("---")
    name=st.text_input("Full Name")
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
    st.write(f"User: {st.session_state['username']}")
    img_file_buffer=st.camera_input("Capture")
    
    if img_file_buffer:
        image=Image.open(img_file_buffer)
        img_array=np.array(image)
        detector=FER(mtcnn=True)
        try:
            emotion, score=detector.top_emotion(img_array)
            st.image(image)
            st.write(f"Emotion: {str(emotion).upper()} ({score*100:.1f}%)")
            
            if st.button("Save"):
                all_data=load_data(DATA_DB)
                user_data=all_data.get(st.session_state['username'], [])
                user_data.append({
                    "date": str(datetime.now()),
                    "type": "face_scan",
                    "emotion": emotion,
                    "score": score,
                    "summary": "N/A"
                })
                all_data[st.session_state['username']]=user_data
                save_data(all_data, DATA_DB)
                st.success("Saved")
        except Exception as e:
            st.error(f"Error: {e}")

def chatbot_page():
    st.title("AI Therapist")
    st.write(f"Hi {st.session_state['username']}! How was your day today?")
    api_key=st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        api_key=st.text_input("API Key", type="password")
    
    if api_key:
        model=init_gemini(api_key)
        if "messages" not in st.session_state:
            st.session_state.messages=[]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type here..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            user_info=st.session_state['user_info']
            context=f"Role: Therapist. Client: {user_info['name']}, Age: {user_info['age']}."
            full_prompt=f"{context}\nUser: {prompt}"
            
            try:
                response=model.generate_content(full_prompt)
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
                all_data=load_data(DATA_DB)
                user_data=all_data.get(st.session_state['username'], [])
                user_data.append({
                    "date": str(datetime.now()),
                    "type": "chat_log",
                    "emotion": "N/A", "score": 0,
                    "summary": prompt[:100]
                })
                all_data[st.session_state['username']]=user_data
                save_data(all_data, DATA_DB)
            except Exception as e:
                st.error(f"API Error: {e}")

def robust_bandpass_filter(signal, lowcut=4.0, highcut=45.0, fs=200, order=4):
    nyquist=0.5 * fs
    low=lowcut / nyquist
    high=highcut / nyquist
    b, a=butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)

@st.cache_resource
def load_eeg_model():
    import os
    import tensorflow as tf
    from tensorflow import keras
    import h5py
    import json
    import zipfile
    import tempfile
    
    model_path = "best_eeg_model.keras"
    
    if not os.path.exists(model_path):
        st.warning("Model file not found")
        return None
    
    try:
        st.write("Attempting to extract model weights and rebuild architecture...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_path = os.path.join(tmpdir, "model")
            
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            config_path = os.path.join(extract_path, "config.json")
            weights_path = os.path.join(extract_path, "model.weights.h5")
            
            if not os.path.exists(config_path):
                raise Exception("Config file not found in model archive")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            st.write("Building compatible model architecture...")
            
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(800, 62)))
            
            for layer_config in config['config']['layers'][1:]:
                layer_class = layer_config['class_name']
                layer_cfg = layer_config['config']
                
                if layer_class == 'Conv1D':
                    model.add(keras.layers.Conv1D(
                        filters=layer_cfg.get('filters', 64),
                        kernel_size=layer_cfg.get('kernel_size', [3])[0],
                        activation=layer_cfg.get('activation', 'relu'),
                        padding=layer_cfg.get('padding', 'valid')
                    ))
                elif layer_class == 'MaxPooling1D':
                    model.add(keras.layers.MaxPooling1D(
                        pool_size=layer_cfg.get('pool_size', [2])[0]
                    ))
                elif layer_class == 'LSTM':
                    model.add(keras.layers.LSTM(
                        units=layer_cfg.get('units', 128),
                        return_sequences=layer_cfg.get('return_sequences', False)
                    ))
                elif layer_class == 'Dropout':
                    model.add(keras.layers.Dropout(
                        rate=layer_cfg.get('rate', 0.5)
                    ))
                elif layer_class == 'Dense':
                    model.add(keras.layers.Dense(
                        units=layer_cfg.get('units', 7),
                        activation=layer_cfg.get('activation', None)
                    ))
                elif layer_class == 'Activation':
                    model.add(keras.layers.Activation(
                        layer_cfg.get('activation', 'softmax')
                    ))
            
            if os.path.exists(weights_path):
                st.write("Loading weights...")
                try:
                    model.load_weights(weights_path)
                    st.success("Model rebuilt and weights loaded successfully!")
                    return model
                except Exception as e:
                    st.warning(f"Weight loading failed: {str(e)[:100]}")
            else:
                st.warning("Weights file not found, using random initialization")
            
            return model
            
    except Exception as e:
        st.error(f"Model rebuild failed: {str(e)}")
        st.write("Attempting fallback: building default CNN-LSTM architecture...")
        
        try:
            model = keras.Sequential([
                keras.layers.Input(shape=(800, 62)),
                keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
                keras.layers.MaxPooling1D(2),
                keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
                keras.layers.MaxPooling1D(2),
                keras.layers.LSTM(128, return_sequences=False),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(7, activation='softmax')
            ])
            st.success("Using default CNN-LSTM architecture (no pretrained weights)")
            return model
        except Exception as e2:
            st.error(f"Fallback failed: {str(e2)}")
    
    return None

def eeg_page():
    st.title("EEG Emotion Recognition")
    
    with st.expander("Model Loading Status", expanded=False):
        model = load_eeg_model()
    
    if model is None:
        model = load_eeg_model()
    
    if not model:
        st.info("Running in demo mode with simulated predictions.")
    else:
        st.success("Model ready! Input shape: (800, 62)")
    
    if st.button("Start Live Simulation"):
        col_graph, col_pred=st.columns([2, 1])
        status=st.empty()
        bar=st.progress(0)
        
        emotion_labels=['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        for i in range(10):
            raw_signal=np.random.normal(0, 1, (800, 62))
            
            filtered=robust_bandpass_filter(raw_signal)
            processed=(filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
            
            input_batch=np.expand_dims(processed, axis=0)
            
            if model:
                preds=model.predict(input_batch, verbose=0)[0]
            else:
                preds=np.random.dirichlet(np.ones(7))
            
            top_idx=np.argmax(preds)
            top_emotion=emotion_labels[top_idx]
            top_conf=preds[top_idx]
            
            with col_graph:
                st.line_chart(processed[:100, 0], height=475)
                
            with col_pred:
                st.metric("Detected", top_emotion, f"{top_conf*100:.1f}%")
                df_probs=pd.DataFrame({"Emotion": emotion_labels, "Prob": preds})
                st.bar_chart(df_probs.set_index("Emotion"))

            status.text(f"Processing batch {i+1}/10...")
            bar.progress((i + 1) * 10)
            time.sleep(0.5) 
            
        status.success("Session Complete")

def ecg_page():
    st.title("ECG Emotion Recognition")
    st.info("ECG tracking feature coming soon. This will monitor heart rate variability to detect stress and emotional states.")
    
    if st.button("Start ECG Simulation"):
        progress_bar=st.progress(0)
        status_text=st.empty()
        chart_placeholder=st.empty()
        
        for i in range(100):
            ecg_signal=np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.normal(0, 0.1, 200)
            chart_placeholder.line_chart(ecg_signal)
            status_text.text(f"Recording: {i+1}%")
            progress_bar.progress(i + 1)
            time.sleep(0.05)
        
        st.success("ECG recording complete")
        st.metric("Average Heart Rate", "72 BPM")
        st.metric("Stress Level", "Low")

def dashboard_page():
    user_info=st.session_state['user_info']
    st.title(f"Hi, {user_info['name']}!")
    st.markdown(f"**Occupation:** {user_info.get('occupation', 'N/A')} | **Base Mood:** {user_info.get('base_mood', 'Neutral')}")
    st.divider()

    all_data=load_data(DATA_DB)
    history=all_data.get(st.session_state['username'], [])
    
    col1, col2, col3=st.columns(3)
    
    total_entries=len(history)
    last_checkin=history[-1]['date'][:10] if history else "No Data"
    
    if history:
        df=pd.DataFrame(history)
        df['date']=pd.to_datetime(df['date'])
        
        if 'emotion' in df.columns and not df['emotion'].empty:
            mode_result=df['emotion'].mode()
            if not mode_result.empty:
                top_emotion=mode_result[0]
            else:
                top_emotion="N/A"
        else:
            top_emotion="N/A"
    else:
        top_emotion="N/A"
        df=pd.DataFrame()

    with col1:
        st.metric("Total Check-ins", total_entries)
    with col2:
        st.metric("Last Check-in", last_checkin)
    with col3:
        st.metric("Dominant Emotion", str(top_emotion).title())

    if not df.empty and 'score' in df.columns and 'emotion' in df.columns:
        st.subheader("Your Emotional Trends")
        
        tab1, tab2=st.tabs(["Timeline", "Distribution"])
        
        with tab1:
            fig_line=px.line(
                df, 
                x='date', 
                y='score', 
                color='emotion', 
                markers=True,
                title="Emotion Intensity Over Time",
                labels={"score": "Confidence Score", "date": "Date"}
            )
            st.plotly_chart(fig_line, width='stretch')
            
        with tab2:
            fig_pie=px.pie(
                df, 
                names='emotion', 
                title="Emotion Distribution",
                hole=0.4
            )
            st.plotly_chart(fig_pie, width='stretch')

    with st.expander("View Personal Details"):
        safe_info=user_info.copy()
        if 'password' in safe_info:
            del safe_info['password']
        df_info=pd.DataFrame(safe_info.items(), columns=['Field', 'Value'])
        df_info['Value']=df_info['Value'].astype(str)
        st.table(df_info)

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in']=False

    if not st.session_state['logged_in']:
        menu=st.sidebar.radio("Menu", ["Login", "Sign Up"])
        if menu == "Login": login_page()
        else: signup_page()
    else:
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
            st.title(f"Welcome, {st.session_state['user_info']['name']}")
            st.write("---")
            menu=st.radio("Menu", ["Dashboard", "AI Therapist", "Face Scan", "EEG", "ECG", "Trends"])
            
            st.write("---")
            if st.button("Logout"):
                st.session_state['logged_in']=False
                st.rerun()

        if menu == "Dashboard":
            dashboard_page()
        elif menu == "AI Therapist":
            chatbot_page()
        elif menu == "Face Scan":
            face_detection_page()
        elif menu == "EEG":
            eeg_page()
        elif menu == "ECG":
            ecg_page()
        elif menu == "Trends":
            st.title("Advanced Analytics")
            st.info("Visit the Dashboard for your current summary graphs.")

if __name__ == "__main__":
    main()





