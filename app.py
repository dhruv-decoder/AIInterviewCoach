import streamlit as st
import os
from main import create_mock_interview

def generate_mock_interview_with_updates(role, experience, additional_details, interview_type):
    yield "Generating mock interview..."
    for update in create_mock_interview(role, experience, additional_details, interview_type, "output"):
        yield update

st.set_page_config(page_title="AIInterviewCoach", page_icon="🎥", layout="wide")

st.title("🎬 AIInterviewCoach")
st.subheader("Generate Custom Mock Interview Videos with AI")
st.write(""" Welcome to AIInterviewCoach! This tool uses advanced AI to create personalized mock interview videos. Perfect for practice, these videos simulate real interview scenarios tailored to your role and experience level. Get ready to ace your next interview! 🚀 """)

# User Inputs
col1, col2 = st.columns(2)
with col1:
    role = st.text_input("🎨 Job Role (e.g., Data Analyst, UX Designer)", "Data Analyst")
    experience = st.selectbox("🌟 Experience Level", ["Entry-level", "Mid-level", "Senior", "Executive"])
with col2:
    interview_type = st.selectbox("🎭 Interview Scenario", ["Standard", "Behavioral", "Technical", "Case Study", "Successful - You're hired!"])
    additional_details = st.text_area("✨ Additional Details", "Proficient in SQL, Python, and Tableau. The company is a growing startup in the e-commerce sector.")

# Create Button
if st.button("🎥 Generate Mock Interview"):
    progress_generator = generate_mock_interview_with_updates(role, experience, additional_details, interview_type)
    progress_text = progress_generator.__next__()
    with st.spinner(progress_text):
        while True:
            try:
                progress_text = progress_generator.__next__()
                st.write(progress_text)
            except StopIteration:
                break

    video_path = os.path.join("output", "mock_interview.mp4")
    if os.path.exists(video_path):
        st.success("🌟 Your mock interview is ready!")
        st.video(video_path)
        st.download_button(
            label="📥 Download Video",
            data=open(video_path, 'rb').read(),
            file_name="my_mock_interview.mp4",
            mime="video/mp4"
        )
    else:
        st.error("❌ An error occurred while generating the mock interview video.")

st.sidebar.title("🚀 About AIInterviewCoach")
st.sidebar.info(""" AIInterviewCoach is an innovative tool that leverages AI to create realistic mock interview videos.   🔧 Features: - Custom scenarios - Role-specific questions - AI-generated visuals - Professional voice-overs  🌈 Created by:  Dhruv Tibarewal 🔗 GitHub: dhruv-decoder """)

st.sidebar.title("🎨 How It Works")
st.sidebar.write(""" 1. 🤖 GPT-3.5 crafts your interview script
                  2. 🎙️ Google TTS brings it to life 
                 3. 🖼️ DALL-E generates relevant imagery 
                 4. 🎬 Python magic assembles your video! """)

st.sidebar.title("💡 Tips")
st.sidebar.success(""" - Be specific about your role - Share unique skills - Try different scenarios - Practice, iterate, succeed! 🌟 """)