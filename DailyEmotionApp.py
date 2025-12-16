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

st.set_page_config(page_title="EmoTrack AI", layout="wide")

# --- GOOGLE SHEETS CONNECTION ---

SHEET_MAPPING = {'users': 'UserDB', 'data': 'DataDB'}

def get_connector():
    return st.connection("gsheets", type=GSheetsConnection)

def load_data(data_type):
    conn = get_connector()
    worksheet_name = SHEET_MAPPING.get(data_type)
    try:
        df = conn.read(worksheet=worksheet_name, ttl=0)
        if data_type == 'users':
            if df.empty: return {}
            df = df.set_index("username")
            return df.to_dict(orient="index")
        elif data_type == 'data':
            if df.empty: return {}
            grouped = df.groupby("username")
            result = {}
            for user, group in grouped:
                records = group.drop(columns=["username"]).to_dict(orient="records")
                result[user] = records
            return result
    except Exception:
        return {}

def save_data(data, data_type):
    conn = get_connector()
    worksheet_name = SHEET_MAPPING.get(data_type)
    if data_type == 'users':
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.name = 'username'
        df.reset_index(inplace=True)
        conn.update(worksheet=worksheet_name, data=df)
    elif data_type == 'data':
        all_records = []
        for user, entries in data.items():
            for entry in entries:
                entry['username'] = user
                all_records.append(entry)
        df = pd.DataFrame(all_records)
        conn.update(worksheet=worksheet_name, data=df)

USER_DB = 'users'
DATA_DB = 'data'

# --- GEMINI AI ---

def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

# --- PAGES ---

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = load_data(USER_DB)
        if username in users and str(users[username]['password']) == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['user_info'] = users[username]
            st.rerun()
        else:
            st.error("Invalid credentials")

def signup_page():
    st.title("Sign Up")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    st.markdown("---")
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1)
    occupation = st.text_input("Occupation")
    mood = st.selectbox("Current Mood", ["Happy", "Sad", "Anxious", "Neutral", "Stressed"])
    health = st.text_area("Health Conditions")
    
    if st.button("Create Account"):
        users = load_data(USER_DB)
        if new_user in users:
            st.error("User exists")
        else:
            users[new_user] = {
                'password': new_pass, 'name': name, 'age': age,
                'occupation': occupation, 'base_mood': mood,
                'health': health, 'joined': str(datetime.now())
            }
            save_data(users, USER_DB)
            st.success("Created! Go to Login.")

def face_detection_page():
    st.title("Real-Time Emotion Detection")
    st.write(f"User: {st.session_state['username']}")
    img_file_buffer = st.camera_input("Capture")
    
    if img_file_buffer:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)
        detector = FER(mtcnn=True)
        try:
            emotion, score = detector.top_emotion(img_array)
            st.image(image)
            st.write(f"Emotion: {str(emotion).upper()} ({score*100:.1f}%)")
            
            if st.button("Save"):
                all_data = load_data(DATA_DB)
                user_data = all_data.get(st.session_state['username'], [])
                user_data.append({
                    "date": str(datetime.now()),
                    "type": "face_scan",
                    "emotion": emotion,
                    "score": score,
                    "summary": "N/A"
                })
                all_data[st.session_state['username']] = user_data
                save_data(all_data, DATA_DB)
                st.success("Saved")
        except Exception as e:
            st.error(f"Error: {e}")

def chatbot_page():
    st.title("AI Therapist")
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("API Key", type="password")
    
    if api_key:
        model = init_gemini(api_key)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type here..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            user_info = st.session_state['user_info']
            context = f"Role: Therapist. Client: {user_info['name']}, Age: {user_info['age']}."
            full_prompt = f"{context}\nUser: {prompt}"
            
            try:
                response = model.generate_content(full_prompt)
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
                all_data = load_data(DATA_DB)
                user_data = all_data.get(st.session_state['username'], [])
                user_data.append({
                    "date": str(datetime.now()),
                    "type": "chat_log",
                    "emotion": "N/A", "score": 0,
                    "summary": prompt[:100]
                })
                all_data[st.session_state['username']] = user_data
                save_data(all_data, DATA_DB)
            except Exception as e:
                st.error(f"API Error: {e}")

def dashboard_page():
    # --- 1. Welcome Header ---
    user_info = st.session_state['user_info']
    st.title(f"Hi, {user_info['name']}!")
    st.markdown(f"**Occupation:** {user_info.get('occupation', 'N/A')} | **Base Mood:** {user_info.get('base_mood', 'Neutral')}")
    st.divider()

    # --- 2. Load & Process Data ---
    all_data = load_data(DATA_DB)
    history = all_data.get(st.session_state['username'], [])
    
    # --- 3. Summary Metrics ---
    col1, col2, col3 = st.columns(3)
    
    total_entries = len(history)
    last_checkin = history[-1]['date'][:10] if history else "No Data"
    
    # Calculate most frequent emotion
    if history:
        df = pd.DataFrame(history)
        # Clean date format
        df['date'] = pd.to_datetime(df['date'])
        
        # Dominant Emotion
        if 'emotion' in df.columns:
            top_emotion = df['emotion'].mode()[0]
        else:
            top_emotion = "N/A"
    else:
        top_emotion = "N/A"
        df = pd.DataFrame()

    with col1:
        st.metric("Total Check-ins", total_entries)
    with col2:
        st.metric("Last Check-in", last_checkin)
    with col3:
        st.metric("Dominant Emotion", str(top_emotion).title())

    # --- 4. Interactive Graphs ---
    if not df.empty and 'score' in df.columns and 'emotion' in df.columns:
        st.subheader("Your Emotional Trends")
        
        tab1, tab2 = st.tabs(["Timeline", "Distribution"])
        
        with tab1:
            # Line Chart: Emotion Score over Time
            fig_line = px.line(
                df, 
                x='date', 
                y='score', 
                color='emotion', 
                markers=True,
                title="Emotion Intensity Over Time",
                labels={"score": "Confidence Score", "date": "Date"}
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
        with tab2:
            # Pie Chart: Emotion Breakdown
            fig_pie = px.pie(
                df, 
                names='emotion', 
                title="Emotion Distribution",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- 5. Secure Profile View (Password Hidden) ---
    with st.expander("View Personal Details"):
        # Create a copy to avoid modifying the actual session state
        safe_info = user_info.copy()
        if 'password' in safe_info:
            del safe_info['password']  # Remove password from display
        st.table(pd.DataFrame(safe_info.items(), columns=['Field', 'Value']))

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        menu = st.sidebar.radio("Menu", ["Login", "Sign Up"])
        if menu == "Login": login_page()
        else: signup_page()
    else:
        # --- Sidebar ---
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100) # Placeholder avatar
            st.title(f"Welcome,\n{st.session_state['user_info']['name']}")
            st.write("---")
            menu = st.radio("Menu", ["Dashboard", "AI Therapist", "Face Scan", "EEG/ECG", "Trends"])
            
            st.write("---")
            if st.button("Logout"):
                st.session_state['logged_in'] = False
                st.rerun()

        # --- Page Routing ---
        if menu == "Dashboard":
            dashboard_page()
        elif menu == "AI Therapist":
            chatbot_page()
        elif menu == "Face Scan":
            face_detection_page()
        elif menu == "EEG/ECG":
            st.title("EEG / ECG Tracker")
            st.info("Hardware integration required. Please connect your BCI device.")
        elif menu == "📈 Trends":
            st.title("📈 Advanced Analytics")
            st.info("Visit the Dashboard for your current summary graphs.")

if __name__ == "__main__":

    main()




