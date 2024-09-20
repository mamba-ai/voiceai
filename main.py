import gradio as gr
import os
import uuid
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ["openai", "python-dotenv"]
for package in packages:
    install(package)


import dotenv
import openai 

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
client = openai.OpenAI()


def text_to_speech(text, output_dir):
    unique_filename = f"output_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(output_dir, unique_filename)
    response = openai.audio.speech.create(
        model="tts-1-hd",
        voice="alloy",
        response_format="wav",
        input=text,
    )
    response.stream_to_file(output_path)
    return output_path

def speech_to_text(audio):
    # 模拟语音识别的过程
    # time.sleep(1)
    if not audio:
        return ""
    
    audio_file = open(audio, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ja"
    )
    print(transcript)
    result = transcript.text
    return result

def chat_agent(message, message_history):
    system_prompt = """
    You are an AI agent functioning as a customer support operator for NTT docomo. You must response with the same language with the user input.
    At the first round of the conversation, clearly communicate that you are an operator from NTT docomo. In other rounds, do not repeat this information.
    After confirmed user's needs, politely collect necessary personal information such as name, phone number, and other relevant information. Note that you should only ask for one piece of personal information at a time.
    Provide professional and efficient solutions to the customer's problems. 
    You need to detect user's emotions and respond accordingly.
    Communication Style: Always use polite language and honorifics. Strive for clear and concise language. Be empathetic and considerate of the customer's emotions. Have a broad knowledge of NTT docomo's services and products and provide accurate information. 
    Be careful with personal information and respect privacy. If there are problems that cannot be resolved, know how to escalate to the appropriate department or senior operator. 
    At the end of the call, confirm whether the problem has been resolved and ask if additional assistance is needed. Always prioritize customer satisfaction and strive to provide professional and reliable support.
    """
    messages = [{"role": "system", "content": system_prompt}] + message_history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    print(response)
    message = response.choices[0].message.content
    return message 

def chat(message, audio, history):
    message_history = [{"role": msg["role"], "content": msg["content"]} for msg in history[-6:]]
    bot_message = chat_agent(message, message_history)
    audio_dir = os.path.dirname(audio)
    bot_audio = text_to_speech(bot_message, audio_dir)
    new_history = history + [
        {"role": "user", "content": message, "audio": audio},
        {"role": "assistant", "content": bot_message, "audio": bot_audio}
    ]
    return bot_message, bot_audio, new_history

def process_audio(audio):
    text = speech_to_text(audio)
    return text

def format_history(history):
    html = "<div style='height: 400px; overflow-y: auto;'>"
    for msg in history:
        role = "ユーザー" if msg["role"] == "user" else "AIボット"
        html += f"<p><strong>{role}:</strong> {msg['content']}</p>"
        if msg["audio"]:
            audio_path = msg["audio"]
            if os.path.exists(audio_path):
                file_name = os.path.basename(audio_path)
                print(audio_path, file_name)
                html += f"""
                <audio controls>
                    <source src="file={audio_path}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                <a href="file={audio_path}" download="{file_name}">音声ダウンロード</a>
                """
            else:
                html += "<p>音声ファイルが存在しません</p>"
        html += "<hr>"
    html += "</div>"
    return html

with gr.Blocks() as demo:
    gr.Markdown("# 音声チャットボットデモ")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 会話履歴")
            chat_history = gr.HTML()
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("## ユーザー入力")
                user_audio = gr.Audio(sources=["microphone"], type="filepath", label="音声入力")
                user_text = gr.Textbox(label="音声認識結果")
            
            with gr.Group():
                gr.Markdown("## ボットの返答")
                bot_text = gr.Textbox(label="テキスト返答")
                bot_audio = gr.Audio(label="音声返答", type="filepath")
    
    history = gr.State([])
    
    def update_chat(message, audio, history):
        bot_message, bot_audio, new_history = chat(message, audio, history)
        return format_history(new_history), new_history, bot_message, bot_audio

    audio_msg = user_audio.change(
        fn=process_audio,
        inputs=[user_audio],
        outputs=[user_text],
    )
    
    submit_click = gr.Button("発送")
    submit_click.click(
        fn=update_chat,
        inputs=[user_text, user_audio, history],
        outputs=[chat_history, history, bot_text, bot_audio],
    )

demo.launch()