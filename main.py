import os
import streamlit as st
import openai
import requests
from PIL import Image
import cv2
import numpy as np
import moviepy.editor as mp
from gtts import gTTS
import io
from PIL import Image, ImageDraw, ImageFont
import textwrap

openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
huggingface_api_key = st.secrets["general"]["HUGGINGFACE_API_KEY"]
# Set up OpenAI API key
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

    print("Transcript saved to 'script.txt'")
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
    print(f"Full interview audio saved to {output_file}")

def generate_image(payload, api_key):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"API Error: {response.status_code} - {response.text}")
        return None
    return response.content

def save_images(prompts, output_folder, api_key):
    custom_instruction = ' Image should be realistic, clear, and appropriate for a professional setting. High quality, 4K resolution.'
    image_paths = []
    os.makedirs(output_folder, exist_ok=True)

    for i, prompt in enumerate(prompts):
        full_prompt = prompt + custom_instruction
        image_bytes = generate_image({"inputs": full_prompt}, api_key)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        image_path = os.path.join(output_folder, f"image{i}.jpg")
        cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print(f"Image {i+1} out of {len(prompts)} saved at {image_path}.")
        image_paths.append(image_path)
    return image_paths

def create_image_with_text(text, background_img, output_file):
    # Open background image and resize
    img = Image.open(background_img)
    img = img.resize((1280, 720))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\Arial.ttf", 36)  # Windows
    except OSError:
        font = ImageFont.load_default()

    # Wrap text
    lines = textwrap.wrap(text, width=40)
    test_text = 'A'
    _, _, _, text_height = font.getbbox(test_text)
    line_height = text_height + 5  # Add a little extra space between lines
    total_height = line_height * len(lines)


    # Calculate starting position
    x = 50  # Left-aligned with padding
    y = 600 - total_height  # Adjusted to accommodate all lines

    # Add a semi-transparent background for better text visibility
    draw.rectangle((0, y - 20, 1280, 720), fill=(0, 0, 0, 128))

    # Draw each line of text
    for line in lines:
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += line_height

    img.save(output_file)

def create_video_with_images_and_audio(images, audio_file, script, output_video='output/interview.mp4'):
    clips = []
    current_background = 0
    clip_duration = 5  # Default duration, but we'll adjust based on audio

    # Load and analyze the audio file
    audio_clip = mp.AudioFileClip(audio_file)
    total_audio_duration = audio_clip.duration
    duration_per_clip = total_audio_duration / len(script)

    for line in script:
        # Select background image (cycle through the list)
        background = backgrounds[current_background]
        current_background = (current_background + 1) % len(backgrounds)
        
        # Create image with text
        img_path = f'temp_image_{len(clips)}.jpg'
        create_image_with_text(line, background, img_path)
        
        # Set clip duration based on audio length
        clip = mp.ImageClip(img_path).set_duration(duration_per_clip)
        clips.append(clip)
        
        # Clean up temporary image file
        os.remove(img_path)
    
    # Simple concatenation without transitions
    video = mp.concatenate_videoclips(clips)
    
    # Initialize last_clip with None
    last_clip = None

    # Ensure video duration matches audio
    if video.duration < total_audio_duration:
        # Extend the last frame to match audio length
        last_frame = video.get_frame(video.duration - 0.1)
        last_clip = mp.ImageClip(last_frame).set_duration(total_audio_duration - video.duration)
        video = mp.concatenate_videoclips([video, last_clip])
    elif video.duration > total_audio_duration:
        # Trim video to match audio length
        video = video.set_duration(total_audio_duration)
    
    # Set the video's audio
    final_clip = video.set_audio(audio_clip)
    
    # Write the final video file
    final_clip.write_videofile(output_video, codec='libx264', fps=24)
    
    # Close all clips to free up resources
    for clip in clips + [video, last_clip, audio_clip, final_clip]:
        try:
            clip.close()
        except:
            pass  # Some clips might not be defined or already closed
 


def create_mock_interview(role, experience, additional_details, interview_type, output_folder='output'):
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Generate Interview Script
    yield "Generating interview transcript..."
    transcript = generate_interview_transcript(role, experience, additional_details, interview_type)

    # Step 2: Generate Audio
    yield "Generating audio..."
    audio_file = os.path.join(output_folder, 'interview_audio.mp3')
    generate_audio(lang='en', output_file=audio_file)

    # Step 3: Generate Images
    yield "Generating images..."
    prompts = [
        f"A professional interviewer in a modern office setting, interviewing for a {role} position.",
        f"A confident candidate being interviewed for a {experience} {role} job, in a corporate environment.",
        "Large screens displaying data charts, graphs, and KPIs in a high-tech office.",
        f"A desk with a laptop showing {role}-related tasks, in a stylish workspace.",
        "A team meeting in a glass-walled conference room, discussing strategies."
    ]
    images = save_images(prompts, output_folder, huggingface_api_key)
    
    global backgrounds
    backgrounds = images  # Use the generated images as backgrounds

    # Step 4: Create Video
    yield "Creating video..."
    script_lines = [line.split(':', 1)[1].strip() for line in transcript.split('\n') if ':' in line]
    output_video = os.path.join(output_folder, 'mock_interview.mp4')
    create_video_with_images_and_audio(images, audio_file, script_lines, output_video)

    yield f"Mock interview video for {experience} {role} position created at {output_video}"















