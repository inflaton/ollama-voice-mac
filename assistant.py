import base64
from io import BytesIO
import json
import re

import requests
import pvporcupine
import pyaudio
import torch
import pyttsx3
import struct
import os
import cv2
from collections import deque
from datetime import datetime
import PIL.Image
import google.generativeai as gemini
from google.generativeai.types.generation_types import BlockedPromptException
from dotenv import load_dotenv
from pathlib import Path
import whisper
import yaml
import numpy as np


def record_audio(stream, rate, frame_length, record_seconds):
    print("Recording...")
    frames = []
    for _ in range(0, int(rate / frame_length * record_seconds)):
        try:
            data = stream.read(frame_length, exception_on_overflow=False)
            frames.append(data)
        except IOError as e:
            if e.errno == pyaudio.paInputOverflowed:
                # Handling overflow
                continue  # Proceed to the next frame
    print("Recording stopped.")
    return b"".join(frames)


def speech_to_text(model, config, audio_data):
    waveform = np.frombuffer(audio_data, np.int16).astype(np.float32) * (1 / 32768.0)
    transcript = model.transcribe(
        waveform,
        language=config.whisperRecognition.lang,
        fp16=torch.cuda.is_available(),
    )
    text = transcript["text"]

    print("\nMe:\n", text.strip())
    return text


def text_to_speech(tts, text):
    print("\nAI:\n", text.strip())
    tts.say(text)
    tts.runAndWait()


def wrap_text(text, line_length):
    """Function to wrap text to the specified length."""
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 > line_length:
            lines.append(current_line)
            current_line = word
        else:
            current_line += " " + word

    lines.append(current_line)  # Adding the last line
    return lines


def add_text_to_frame(frame, text):
    # Wrapping text every 70 characters
    wrapped_text = wrap_text(text, 70)

    # Getting the height and width of the frame
    height, width = frame.shape[:2]

    # Setting the text font and size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # Increasing font size
    color = (255, 255, 255)  # White color
    outline_color = (0, 0, 0)  # Outline color (black)
    thickness = 2
    outline_thickness = 4  # Outline thickness
    line_type = cv2.LINE_AA

    # Adding each line of text to the image
    for i, line in enumerate(wrapped_text):
        position = (10, 30 + i * 30)  # Adjusting the position of each line (larger gap)

        # Drawing the outline of the text
        cv2.putText(
            frame,
            line,
            position,
            font,
            font_scale,
            outline_color,
            outline_thickness,
            line_type,
        )

        # Drawing the text
        cv2.putText(
            frame, line, position, font, font_scale, color, thickness, line_type
        )


def save_frame(frame, filename, directory="./frames"):
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Creating the path for the filename
    filepath = os.path.join(directory, filename)
    # Saving the frame
    cv2.imwrite(filepath, frame)


def save_temp_frame(frame, filename, directory="./temp"):
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Creating the path for the filename
    filepath = os.path.join(directory, filename)
    # Saving the frame
    cv2.imwrite(filepath, frame)
    return filepath  # Returning the path of the saved file


def ask_ollama_with_image(
    image_path_or_url,
    model="llava",
    url="http://localhost:11434/api/generate",
    user_input="What is in this picture?",
):
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        buffered = BytesIO(response.content)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        with open(image_path_or_url, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = f"Please respond to the following message, using no more than 20 words. Message: {user_input}"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [image_base64],
    }

    # Send POST request
    response = requests.post(url, data=json.dumps(payload))

    # Print response
    print(response.text)
    response_dict = json.loads(response.text)

    return (
        response_dict["response"]
        if "response" in response_dict
        else response_dict["error"]
    )


def send_frame_with_text_to_llm(
    frame, previous_texts, timestamp, user_input, config, llm_model_type
):
    temp_file_path = save_temp_frame(frame, "temp.jpg")

    if llm_model_type == "ollama":
        return ask_ollama_with_image(
            temp_file_path,
            model=config.ollama.model,
            url=config.ollama.url,
            user_input=user_input,
        )

    img = PIL.Image.open(temp_file_path)

    # Combining past texts as context
    context = " ".join(previous_texts)

    # Adding system message
    system_message = "System Message - Your identity: Gemini, you are a smart, kind, and helpful AI assistant."

    # Initializing Gemini model
    gemini.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = gemini.GenerativeModel(config.gemini.model)

    # Sending image and text instructions to the model
    prompt = f"{system_message}\nGiven the context: {context} and the current time: {timestamp}, please respond to the following message without repeating the context, using no more than 20 words. Message: {user_input}"

    try:
        response = model.generate_content([prompt, img], stream=False)
        response.resolve()
        # Returning the generated text
        return response.text
    except BlockedPromptException as e:
        print(
            "AI response was blocked due to safety concerns. Please try a different input."
        )
        return "AI response was blocked due to safety concerns."


def init_config():
    class Inst:
        pass

    with open("./config/config.yaml", encoding="utf-8") as data:
        configYaml = yaml.safe_load(data)

    config = Inst()
    config.messages = Inst()
    config.messages.loadingModel = configYaml["messages"]["loadingModel"]
    config.messages.pressSpace = configYaml["messages"]["pressSpace"]
    config.messages.noAudioInput = configYaml["messages"]["noAudioInput"]

    config.conversation = Inst()
    config.conversation.greeting = configYaml["conversation"]["greeting"]

    config.picoVoice = Inst()
    config.picoVoice.keywordPath = configYaml["picoVoice"]["keywordPath"]

    config.ollama = Inst()
    config.ollama.url = configYaml["ollama"]["url"]
    config.ollama.model = configYaml["ollama"]["model"]

    config.whisperRecognition = Inst()
    config.whisperRecognition.modelPath = configYaml["whisperRecognition"]["modelPath"]
    config.whisperRecognition.lang = configYaml["whisperRecognition"]["lang"]

    return config


def main():
    print("workding dir:", Path.cwd())
    load_dotenv(override=False)

    config = init_config()
    print(config.messages.loadingModel)

    model = whisper.load_model(config.whisperRecognition.modelPath)

    # Loading the access key and keyword path from environment variables
    access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
    keyword_path = config.picoVoice.keywordPath

    # Creating a Porcupine instance
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])

    # Initializing PyAudio
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )

    tts = pyttsx3.init("nsss")
    tts.setProperty("rate", tts.getProperty("rate") - 20)

    exit_pattern = re.compile(r"(bye.*){2}", re.IGNORECASE)

    llm_model_type = os.environ.get("LLM_MODEL_TYPE")
    print("LLM model type:", llm_model_type)

    try:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise IOError("Could not open the camera.")

        previous_texts = deque(maxlen=5)

        text_to_speech(tts, config.conversation.greeting)

        while True:
            try:
                # Reading audio data from PyAudio stream
                pcm = audio_stream.read(
                    porcupine.frame_length, exception_on_overflow=False
                )
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                # Detecting wake word using Porcupine
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:  # If wake word is detected
                    print("Wake word detected!")

                    # Recording voice input and converting it to text
                    audio_data = record_audio(
                        audio_stream,
                        porcupine.sample_rate,
                        porcupine.frame_length,
                        3,
                    )
                    user_input = speech_to_text(model, config, audio_data)

                    # Processing if there is voice input
                    if user_input:  # If there is voice input
                        if re.search(exit_pattern, user_input):
                            break

                        # Image processing and AI response code
                        success, frame = video.read()  # Reading a frame from the camera
                        if not success:
                            print("Failed to read frame.")
                            break  # Exiting the loop if frame reading fails

                        timestamp = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )  # Getting the current timestamp

                        # Sending frame and user input to Gemini AI model and generating a response
                        generated_text = send_frame_with_text_to_llm(
                            frame,
                            previous_texts,
                            timestamp,
                            user_input,
                            config,
                            llm_model_type,
                        )
                        print(
                            f"Timestamp: {timestamp}, Generated Text: {generated_text}"
                        )

                        # Updating past texts
                        # previous_texts.append(f"[{timestamp}] Message: {user_input}, Generated Text: {generated_text}")
                        previous_texts.append(
                            f"Timestamp: {timestamp}\nUser Message: {user_input}\nYour Response: {generated_text}\n"
                        )

                        # Adding the generated text to the frame
                        text_to_add = (
                            f"{timestamp}: {llm_model_type} - {generated_text}"
                        )
                        add_text_to_frame(frame, text_to_add)

                        # Saving the frame
                        filename = f"{timestamp}.jpg"
                        save_frame(frame, filename)  # Saving as an image

                        text_to_speech(tts, generated_text)

                    else:  # If there is no voice input
                        print("No user input, exiting the loop.")
                        break  # Exiting the loop

            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    print("Input overflow, restarting the stream")
                    if audio_stream.is_active():
                        audio_stream.stop_stream()
                    if not audio_stream.is_stopped():
                        audio_stream.start_stream()
                else:
                    raise e

    finally:
        audio_stream.close()
        pa.terminate()
        porcupine.delete()
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
