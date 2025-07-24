import os
import whisper
import cohere
from gtts import gTTS
from playsound import playsound

COHERE_API_KEY = "cohere api key"  
AUDIO_FILE = "input.mp3"              # audio file path
OUTPUT_AUDIO = "output_response.mp3"        # Output speech file

def transcribe_audio(audio_path):
    print("Loading Whisper model...")
    model = whisper.load_model("base")  
    result = model.transcribe(audio_path)
    return result["text"]

def generate_response(prompt):
    co = cohere.Client(COHERE_API_KEY)
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=100
    )
    return response.generations[0].text

def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)
    playsound(output_file)  

if __name__ == "__main__":
    user_input = transcribe_audio(AUDIO_FILE)
    print(f"You said: {user_input}")

    ai_response = generate_response(user_input)
    print(f"AI says: {ai_response}")

    text_to_speech(ai_response, OUTPUT_AUDIO)