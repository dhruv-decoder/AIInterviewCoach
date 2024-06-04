import os
import streamlit as st
import openai
import requests
from PIL import Image, ImageDraw, ImageFont
import textwrap
import moviepy.editor as mp
from gtts import gTTS
import io
import numpy as np

# API keys
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
huggingface_api_key = st.secrets["general"]["HUGGINGFACE_API_KEY"]
openai.api_key = openai_api_key

def generate_interview_transcript(role, experience, additional_details, interview_type):
    model_engine = "gpt-3.5-turbo"

    if role.lower() == "invalid" or experience.lower() == "invalid":
        return "Sorry, the provided role or experience level seems invalid. Please enter a valid and realistic role and experience level."

    prompt = f"Generate a {interview_type} mock interview script to be used in a video for a {experience} {role} candidate. Incorporate any relevant details like candidate's name, interviewer's name, company details, etc. Keep the transcript concise and focused on the interview conversation. Additional details: {additional_details}"

    messages = [
        {"role": "system", "content": "You are an assistant generating realistic mock interview transcripts to help candidates prepare for interviews. Do not generate any inappropriate or unrealistic content."},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=messages,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    transcript = response.choices[0].message.content
    with open("script.txt", "w", encoding="utf-8") as file:
        file.write(transcript)

    return transcript

def generate_audio(script_file='script.txt', output_file='interview_audio.mp3', lang='en'):
    with open(script_file, 'r') as file:
        script = file.read()

    lines = script.split('\n')
    full_text = ""
    current_speaker = None

    for line in lines:
        if ':' in line:
            speaker, text = line.split(':', 1)
            text = text.strip()
            if not text: continue
            
            if speaker != current_speaker:
                full_text += "... ... ... "
                current_speaker = speaker
            
            full_text += text + ". "

    tts = gTTS(text=full_text, lang=lang, slow=False)
    tts.save(output_file)

def generate_image(prompt, api_key):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code != 200:
        raise Exception(f"Failed to generate image: {response.status_code} - {response.text}")
    return Image.open(io.BytesIO(response.content))

def create_video_with_images_and_audio(images, audio_file, script, output_video='output/interview.mp4'):
    clips = []
    audio_clip = mp.AudioFileClip(audio_file)
    total_audio_duration = audio_clip.duration
    lines = [line for line in script.split('\n') if ':' in line]

    duration_per_clip = total_audio_duration / len(lines)

    for i, line in enumerate(lines):
        speaker, text = line.split(':', 1)
        text = text.strip()

        # Choose the image based on the speaker
        background = images[0] if 'Interviewer' in speaker else images[1]
        
        # Use the image directly without adding text
        clip = mp.ImageClip(np.array(background)).set_duration(duration_per_clip)
        clips.append(clip)

    video = mp.concatenate_videoclips(clips)
    final_clip = video.set_audio(audio_clip)
    final_clip.write_videofile(output_video, codec='libx264', fps=24)

def create_mock_interview(role, experience, additional_details, interview_type, output_folder='output'):
    os.makedirs(output_folder, exist_ok=True)

    transcript = generate_interview_transcript(role, experience, additional_details, interview_type)

    audio_file = os.path.join(output_folder, 'interview_audio.mp3')
    generate_audio(lang='en', output_file=audio_file)

    # Generate images for the interviewer and interviewee
    prompts = [
        "A professional interviewer in a modern office setting, clear and realistic.",
        "A confident candidate being interviewed, in a corporate environment, clear and realistic."
    ]

    images = []
    for prompt in prompts:
        image = generate_image(prompt, huggingface_api_key)
        images.append(image)

    output_video = os.path.join(output_folder, 'mock_interview.mp4')
    create_video_with_images_and_audio(images, audio_file, transcript, output_video)

    yield f"Mock interview video for {experience} {role} position created at {output_video}"

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
