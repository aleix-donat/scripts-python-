import os
import subprocess
import requests
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
import httplib2
import ssl
import google.auth
import google.auth.transport.requests
from google.cloud import texttospeech
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, UnidentifiedImageError
import random
from pydub import AudioSegment
from pydub.utils import which
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import nltk
import time
from google.oauth2 import service_account
import json
import re

# Verify ffmpeg configuration
ffmpeg_path = which("ffmpeg")
if ffmpeg_path:
    print(f"FFmpeg found at {ffmpeg_path}")
else:
    print("FFmpeg not found in PATH")

# Verify ImageMagick configuration
imagemagick_path = which("magick")
if imagemagick_path:
    os.environ['IMAGE_MAGICK_BINARY'] = imagemagick_path
    print(f"ImageMagick found at {imagemagick_path}")
else:
    print("ImageMagick not found in PATH")

# Configure ffmpeg for pydub
AudioSegment.converter = ffmpeg_path
print("FFmpeg Path for pydub:", AudioSegment.converter)

# Configure ffmpeg for moviepy
import moviepy.config as moviepy_config
moviepy_config.change_settings({"FFMPEG_BINARY": ffmpeg_path})
print("FFmpeg Path for moviepy:", moviepy_config.get_setting("FFMPEG_BINARY"))

# Configure ImageMagick in MoviePy
moviepy_config.IMAGEMAGICK_BINARY = os.environ['IMAGE_MAGICK_BINARY']

# Verify ffmpeg can be executed
try:
    result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
    print(result.stdout)
except FileNotFoundError as e:
    print(f"FFmpeg not found: {e}")

# Load environment variables
print("Loading environment variables")
load_dotenv()

# Check if environment variables have been loaded correctly
news_api_key = os.getenv('NEWS_API_KEY')
youtube_api_key = os.getenv('YOUTUBE_API_KEY')
google_custom_search_api_key = os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
search_engine_id = os.getenv('GOOGLE_CUSTOM_SEARCH_CX')
pexels_api_key = os.getenv('PEXELS_API_KEY')
youtube_client_secret_path = os.getenv('YOUTUBE_CLIENT_SECRET_PATH')
openai_api_key = os.getenv('OPENAI_API_KEY')

print("News API Key:", news_api_key)
print("YouTube API Key:", youtube_api_key)
print("Google API Key:", google_custom_search_api_key)
print("Search Engine ID:", search_engine_id)
print("Pexels API Key:", pexels_api_key)
print("YouTube Client Secret Path:", youtube_client_secret_path)
print("OpenAI API Key:", openai_api_key)

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
english_stopwords = set(stopwords.words('english'))
spanish_stopwords = set(stopwords.words('spanish'))

# Current topics in Spanish politics
current_topics = [
    "política española actual",
    "elecciones en España",
    "partidos políticos en España",
    "gobierno de España",
    "crisis política en España"
]

# List of political figures
political_figures = [
    "Pedro Sánchez",
    "Pablo Casado",
    "Santiago Abascal",
    "Albert Rivera",
    "Íñigo Errejón",
    "Pablo Iglesias",
    "Inés Arrimadas",
    "Mariano Rajoy",
    "José Luis Rodríguez Zapatero",
    "Felipe González",
    "Adolfo Suárez"
]

# Get a random topic
def get_random_topic():
    return random.choice(current_topics)

# Fetch current news for a topic
def get_news(api_key, query, language='es'):
    print(f"Fetching current news for topic: {query}")
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&apiKey={api_key}&pageSize=100"
    response = requests.get(url)
    data = response.json()
    print("News API response:", data)
    return data

# Correct translations from English to Spanish
def correct_translation(text):
    translations = {
        'that': 'que', 'and': 'y', 'with': 'con', 'of': 'de',
        'to': 'a', 'in': 'en', 'on': 'en', 'for': 'para', 'is': 'es',
        'was': 'fue', 'were': 'fueron', 'are': 'son', 'be': 'ser', 'have': 'tener',
        'this': 'este', 'which': 'que', 'it': 'eso', 'as': 'como', 'by': 'por'
    }
    pattern = re.compile(r'\b(' + '|'.join(translations.keys()) + r')\b')
    return pattern.sub(lambda x: translations[x.group()], text)

# Clean and correct a sentence
def clean_and_correct_sentence(sentence):
    sentence = re.sub(r'\s+', ' ', sentence)  # Remove multiple spaces
    if not sentence.endswith('.'):
        sentence += '.'
    return sentence

# Remove English words from text
def remove_english_words(text):
    words = text.split()
    words = [word for word in words if word.lower() not in english_stopwords]
    return ' '.join(words)

# Create a script from news articles
def create_script(news_articles, max_duration_seconds=300):
    print("Creating script")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    script = ""
    total_words = 0
    max_words = max_duration_seconds * 2.5  # Approximately 2.5 words per second

    introductions = [
        "En la actualidad,",
        "Hoy,",
        "Noticias recientes:",
        "Lo más destacado:",
        "Últimas noticias:",
        "En el ámbito político,",
        "Para mantenernos informados,"
    ]

    print(f"Total articles found: {len(news_articles.get('articles', []))}")
    for article in news_articles.get('articles', []):
        print(f"Article title: {article.get('title', 'No Title')}")
        content = article.get('content', '') or article.get('description', '') or article.get('title', '')
        if content:
            content = correct_translation(content)
            print(f"Article content: {content[:200]}...")  # Print first 200 characters
            summary = summarizer(content, max_length=150, min_length=40, do_sample=False)
            summary_text = summary[0]['summary_text']
            summary_text = correct_translation(summary_text)
            summary_text = clean_and_correct_sentence(summary_text)
            summary_text = remove_english_words(summary_text)  # Remove English words
            summary_words = summary_text.split()
            total_words += len(summary_words)
            if total_words <= max_words:
                introduction = random.choice(introductions)
                script += f"{introduction} {summary_text}\n"
            else:
                break
    
    # Ensure not to cut off words at the end of the script
    script_words = script.split()
    if len(script_words) > max_words:
        script = " ".join(script_words[:int(max_words)])
    
    return script

# Clean text
def clean_text(text):
    text = re.sub(r'\b(?:{})\b'.format('|'.join(spanish_stopwords)), '', text)
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

# Extract keywords from the script
def extract_keywords(script, num_keywords=20):
    print("Extracting keywords")
    if not script.strip():
        print("The script is empty after summarization. No keywords to extract.")
        return []

    cleaned_script = clean_text(script)
    vectorizer = CountVectorizer(max_df=1, min_df=1, stop_words=list(spanish_stopwords))
    word_count = vectorizer.fit_transform([cleaned_script])
    keywords = np.array(vectorizer.get_feature_names_out())
    counts = word_count.toarray().flatten()
    top_keywords = keywords[counts.argsort()[-num_keywords:]]
    print("Extracted Keywords:", top_keywords)
    return top_keywords

# Search images for a query
def search_images(query, api_key, cx, num_images=5):
    print(f"Searching images for query: {query}")
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={cx}&searchType=image&key={api_key}&num={num_images}"
    response = requests.get(url)
    response.raise_for_status()
    search_results = response.json()
    image_urls = [img["link"] for img in search_results.get("items", [])]
    return image_urls

# Download an image from a URL
def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        print(f"Downloaded image from {url}")
        return img
    except (requests.RequestException, UnidentifiedImageError) as e:
        print(f"Error downloading image from {url}: {e}")
        return None

# Create a thumbnail image
def create_thumbnail(keywords):
    print("Creating thumbnail")
    images = []
    found_images = False

    # Additional keywords to increase the chance of finding relevant images
    additional_keywords = political_figures + ["política", "gobierno", "elecciones", "parlamento", "debate"]
    
    # Convert keywords to list if not already
    if not isinstance(keywords, list):
        keywords = list(keywords)
    
    # Combine original and additional keywords
    search_keywords = keywords + additional_keywords
    
    for keyword in search_keywords:
        image_urls = search_images(keyword, google_custom_search_api_key, search_engine_id, num_images=1)
        for url in image_urls:
            img = download_image(url)
            if img:
                img = img.convert('RGB')
                images.append(img)
                found_images = True
                break
        if found_images:
            break

    if len(images) == 0:
        print("No images found. Exiting.")
        return None

    background_img = images[0].resize((1280, 720), Image.LANCZOS)
    background_img = background_img.filter(ImageFilter.GaussianBlur(5))  # Apply Gaussian blur

    collage = background_img
    draw = ImageDraw.Draw(collage)

    overlay = Image.new('RGBA', collage.size, (0, 0, 0, 180))
    collage = Image.alpha_composite(collage.convert('RGBA'), overlay)

    draw = ImageDraw.Draw(collage)
    try:
        font_large = ImageFont.truetype("impact.ttf", 110)  # Use "Impact" font with larger size
    except IOError:
        font_large = ImageFont.load_default()

    main_text = random.choice([word for word in keywords if word in political_figures or len(word.split()) == 1]).capitalize()  # Select a single keyword and capitalize it

    # Determine the size of the text box
    text_bbox = draw.textbbox((0, 0), main_text, font=font_large)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Create the rectangle with a white background, bevel, and shadow
    padding = 30
    rectangle_x0 = (collage.width - text_width) // 2 - padding
    rectangle_y0 = collage.height - text_height - 90 - padding  # Adjust slightly downward
    rectangle_x1 = (collage.width + text_width) // 2 + padding
    rectangle_y1 = collage.height - 60 + padding  # Adjust slightly downward
    bevel = 10
    shadow_offset = 10

    # Draw the rectangle shadow
    draw.rounded_rectangle([(rectangle_x0 + shadow_offset, rectangle_y0 + shadow_offset), (rectangle_x1 + shadow_offset, rectangle_y1 + shadow_offset)], fill="black", outline=None, radius=bevel)
    
    # Draw the rectangle with bevel
    for i in range(bevel):
        alpha = int(255 * (i / float(bevel)))
        outline_color = (0, 0, 0, alpha)
        draw.rounded_rectangle([(rectangle_x0 + i, rectangle_y0 + i), (rectangle_x1 - i, rectangle_y1 - i)], fill=None, outline=outline_color, width=1, radius=bevel-i)
    
    draw.rounded_rectangle([(rectangle_x0, rectangle_y0), (rectangle_x1, rectangle_y1)], fill="white", outline="black", width=2, radius=bevel)
    
    # Draw the text with outline and shadow
    text_x = (collage.width - text_width) // 2
    text_y = rectangle_y0 + (rectangle_y1 - rectangle_y0 - text_height) // 2 - 5  # Adjust slightly upward
    shadow_offset = 3
    
    # Text shadow
    draw.text((text_x + shadow_offset, text_y + shadow_offset), main_text, font=font_large, fill="black")

    # Text outline
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            draw.text((text_x + dx, text_y + dy), main_text, font=font_large, fill="black")
    
    # Main text
    draw.text((text_x, text_y), main_text, font=font_large, fill="#FFA500")  # Orange-yellow color

    frame_color = tuple(np.random.choice(range(256), 3))
    draw.rectangle([(0, 0), (collage.width, collage.height)], outline=frame_color, width=10)
    
    thumbnail_path = 'thumbnail.jpg'
    collage.convert('RGB').save(thumbnail_path)
    print(f"Thumbnail saved at {thumbnail_path}")
    return thumbnail_path

# Generate metadata for the video
def generate_metadata(script, keywords):
    print("Generating metadata")

    # Filter English words and keep only relevant Spanish ones
    filtered_keywords = [word for word in keywords if word.lower() not in ["the", "is", "and", "to", "of", "with", "by"]]
    
    # Generate a simple title using Spanish keywords
    if len(filtered_keywords) > 3:
        title = f"{filtered_keywords[0]} en {filtered_keywords[1]}: {filtered_keywords[2]} y {filtered_keywords[3]}"
    elif len(filtered_keywords) > 2:
        title = f"{filtered_keywords[0]} en {filtered_keywords[1]}: {filtered_keywords[2]}"
    elif len(filtered_keywords) > 1:
        title = f"{filtered_keywords[0]} en {filtered_keywords[1]}"
    else:
        title = filtered_keywords[0]
    
    # Limit the title length to 70 characters
    if len(title) > 70:
        title = title[:67] + "..."

    # Ensure the first letter of the title is capitalized
    title = title.capitalize()
    
    # Generate an introduction for the description
    description_intro = f"This video covers the highlights of {filtered_keywords[0]}, addressing topics such as {', '.join(filtered_keywords[1:4])}."
    
    # Limit the description to 300 characters
    description = description_intro[:297] + "..." if len(description_intro) > 300 else description_intro
    
    tags = list(filtered_keywords)
    extra_tags = ["news", "current events", "politics", "Spain politics"]
    tags.extend(extra_tags)
    
    return title, description, tags

# Generate voice with Google Cloud Text-to-Speech
def generate_voice_with_google(script, target_duration_seconds=300):
    print("Generating voice with Google Cloud Text-to-Speech")
    
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"Using credentials from {credentials_path}")
    
    if not os.path.exists(credentials_path):
        print(f"Credentials file not found: {credentials_path}")
        return
    
    try:
        with open(credentials_path, 'r') as file:
            credentials_info = json.load(file)
            print(f"Credentials loaded: {credentials_info.keys()}")
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
    except Exception as e:
        print(f"Error loading credentials: {e}")
        return
    
    client = texttospeech.TextToSpeechClient(credentials=credentials)

    max_words = target_duration_seconds * 2.5
    words = script.split()
    if len(words) > max_words:
        script = " ".join(words[:int(max_words)])

    # Split text into chunks of max 5000 characters
    def split_text(text, max_length=5000):
        chunks = []
        current_chunk = ''
        for sentence in text.split('. '):
            if len(current_chunk.encode('utf-8')) + len(sentence.encode('utf-8')) + 1 <= max_length:
                current_chunk += sentence + '. '
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    script_chunks = split_text(script)
    final_audio = AudioSegment.empty()

    for idx, chunk in enumerate(script_chunks):
        synthesis_input = texttospeech.SynthesisInput(text=chunk)

        voice = texttospeech.VoiceSelectionParams(
            language_code="es-ES",
            name="es-ES-Wavenet-B",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        part = f"voice_part_{idx + 1}.mp3"
        with open(part, "wb") as out:
            out.write(response.audio_content)
        print(f"Generated voice part saved as {part}")

        final_audio += AudioSegment.from_mp3(part)

    final_audio_path = "voice.mp3"
    final_audio.export(final_audio_path, format="mp3")
    print(f"Final voice saved as {final_audio_path}")
    return final_audio_path

# Download stock videos from Pexels
def download_pexels_videos(keywords, max_results=5):
    print(f"Downloading stock videos from Pexels for keywords: {keywords}")
    videos = []
    for keyword in keywords:
        query = f"{keyword} política española"
        url = f"https://api.pexels.com/videos/search?query={query}&per_page={max_results}"
        headers = {
            "Authorization": pexels_api_key
        }
        response = requests.get(url, headers=headers)
        print(f"Response for keyword '{keyword}': {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response data for keyword '{keyword}': {data}")
            if 'videos' in data and data['videos']:
                for video in data['videos']:
                    print(f"Checking video: {video['id']} with title: {video.get('title', '')}")
                    video_url = video['video_files'][0]['link']
                    video_response = requests.get(video_url)
                    video_filename = f"{keyword.replace(' ', '_')}_{video['id']}.mp4"
                    with open(video_filename, 'wb') as f:
                        f.write(video_response.content)
                    videos.append(video_filename)
                    print(f"Downloaded stock video for keyword {keyword}")
                    if len(videos) >= max_results:
                        break
            else:
                print(f"No stock videos found for keyword {keyword}. Response data: {data}")
        else:
            print(f"Error fetching stock videos for keyword {keyword}: {response.status_code}")
    
    return videos

# Resize video clips to the same resolution
def resize_clips_to_same_resolution(clips, target_resolution=(1280, 720)):
    resized_clips = []
    for clip in clips:
        if clip.size != target_resolution:
            clip = clip.resize(target_resolution)
        resized_clips.append(clip)
    return resized_clips

# Create the final video
def create_final_video(videos, audio_path, script):
    print("Creating final video")
    
    if not videos:
        print("No videos downloaded. Exiting.")
        return
    
    clips = [VideoFileClip(video) for video in videos if video]
    valid_clips = [clip for clip in clips if clip.duration > 0]
    target_duration = 300  # 5 minutes
    num_clips = len(valid_clips)
    
    if num_clips == 0:
        print("No valid video clips available. Exiting.")
        return
    
    section_duration = target_duration / num_clips
    resized_clips = resize_clips_to_same_resolution(valid_clips)
    final_video = concatenate_videoclips([clip.subclip(0, min(section_duration, clip.duration)) for clip in resized_clips])
    
    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} not found. Exiting.")
        return
    
    audio = AudioSegment.from_file(audio_path)
    audio_duration = len(audio) / 1000

    if final_video.duration > audio_duration:
        final_video = final_video.subclip(0, audio_duration)

    adjusted_audio = audio[:int(final_video.duration * 1000)]
    adjusted_audio_path = "adjusted_voice.mp3"
    adjusted_audio.export(adjusted_audio_path, format="mp3")

    audio_clip = AudioFileClip(adjusted_audio_path)
    final_video = final_video.set_audio(audio_clip)
    final_video.write_videofile("final_video.mp4", codec='libx264', preset='slow', bitrate='192k', audio_codec='aac')
    
    enhance_video_quality('final_video.mp4', 'final_video_enhanced.mp4')

# Enhance video quality
def enhance_video_quality(input_path, output_path):
    command = [
        'ffmpeg', '-i', input_path, '-vf', 'scale=1280:720:flags=lanczos', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-c:a', 'aac', '-b:a', '192k', '-movflags', 'faststart', output_path
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Video quality enhanced and saved at: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error enhancing video quality: {e}")
        return None

# Reprocess video
def reprocesar_video(input_path, output_path):
    command = [
        'ffmpeg', '-i', input_path, '-c:v', 'libx264', '-preset', 'slow', '-crf', '22', '-c:a', 'aac', '-b:a', '192k', '-movflags', 'faststart', output_path
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Video reprocessed and saved at: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error reprocessing video: {e}")
        return None

# Upload video to YouTube
def upload_to_youtube(title, description, tags, video_path, thumbnail_path):
    print("Uploading to YouTube")
    
    if not title or not title.strip():
        raise ValueError("The video title is empty or invalid. Please provide a valid title.")
    
    print(f"Video Title: '{title}'")
    
    if thumbnail_path is None:
        print("Thumbnail creation failed. Exiting.")
        return

    flow = InstalledAppFlow.from_client_secrets_file(
        'youtube_client_secret.json', scopes=["https://www.googleapis.com/auth/youtube.upload"]
    )
    credentials = flow.run_local_server(port=0)

    youtube = build("youtube", "v3", credentials=credentials)

    try:
        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "categoryId": "25"
                },
                "status": {
                    "privacyStatus": "public"
                }
            },
            media_body=MediaFileUpload(video_path)
        )

        response = request.execute()
        video_id = response["id"]
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(thumbnail_path)
        ).execute()
        print("Video uploaded to YouTube")
    except HttpError as e:
        if e.resp.status == 403:
            print("Quota exceeded. Aborting upload.")
        elif e.resp.status == 401:
            print("Authentication error. Please check your credentials.")
        elif e.resp.status == 400:
            print(f"Failed to upload video: {e}. The title or description might be invalid or empty.")
        else:
            print(f"Failed to upload video: {e}")
    except ssl.SSLWantWriteError:
        print("SSL error occurred while uploading the video. Please try again later.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Main function
def main():
    print("Starting process")
    topic = get_random_topic()
    print(f"Selected topic: {topic}")
    
    news = get_news(news_api_key, topic)
    print("News fetched")
    script = create_script(news, max_duration_seconds=300)
    print(f"Script created: {script[:200]}...")  # Print first 200 characters of the script
    keywords = extract_keywords(script)
    if len(keywords) == 0:
        print("No keywords extracted. Exiting.")
        return
    print("Keywords extracted")
    
    pexels_videos = download_pexels_videos(keywords, max_results=10)
    videos = pexels_videos
    
    print(f"Videos downloaded: {videos}")
    if not videos:
        print("No videos were downloaded. Exiting.")
        return
    thumbnail_path = create_thumbnail(keywords)
    if thumbnail_path is None:
        print("Failed to create thumbnail. Exiting.")
        return
    print("Thumbnail created")
    title, description, tags = generate_metadata(script, keywords)
    print(f"Metadata generated\nTitle: {title}\nDescription: {description}\nTags: {tags}")
    audio_path = generate_voice_with_google(script, target_duration_seconds=300)
    if not audio_path:
        print("Voice generation failed. Exiting.")
        return
    print("Voice generated")
    create_final_video(videos, audio_path, script)
    print("Final video created")
    
    reprocesado_video_path = reprocesar_video('final_video_enhanced.mp4', 'final_video_reprocesado.mp4')
    if reprocesado_video_path:
        upload_to_youtube(title, description, tags, reprocesado_video_path, thumbnail_path)
    else:
        print("Failed to reprocess video. Exiting.")

if __name__ == "__main__":
    main()
