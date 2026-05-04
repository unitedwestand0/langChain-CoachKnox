
import streamlit as st
from backend.core import run_llm
import os
import warnings
import logging
from dotenv import load_dotenv

from elevenlabs import stream
from elevenlabs.client import ElevenLabs
import speech_recognition as sr

from streamlit.components.v1 import html


load_dotenv()


# Suppress the Pydantic V1 warning (common with LangChain on Python 3.14)
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.")

# Suppress the noisy __path__ warnings from transformers vision models
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message="Accessing `__path__` from")

# Optional: lower transformers logging level to ERROR to reduce noise
logging.getLogger("transformers").setLevel(logging.ERROR)




st.set_page_config(page_title="MMA AI Coach - Grok Powered", layout="wide")


#st.title("Welcome to LEGEND")
#st.subheader("Welcome to LEGEND")
#st.subheader("I'm Conor McGregor, your AI Boxing Coach!")
#st.caption("I'm designed to help fighters like you improve your skills and knowledge.")

# =========================== 3D CHATBOT MODEL DISPLAY =========================== #

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("")  # Empty space for centering
html("""
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.2.0/model-viewer.min.js"></script>

<model-viewer 
    src="https://files.catbox.moe/ppcgg0.glb"
    alt="Conor McGregor 3D Model"
    auto-rotate 
    camera-controls 
    style="width:100%; height:600px; background:#111;"
    shadow-intensity="1"
    exposure="0.8"
    camera-orbit="45deg 75deg 60m">
</model-viewer>
""", height=650)
# =========================== SIDEBAR WITH COACH KNOX =========================== #

# Sidebar Settings
with st.sidebar:

    # centering his image 
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:

        st.title("Welcome to LEGEND")
        st.subheader("I'm Conor McGregor, your AI Boxing Coach!")
        st.caption("I'm designed to help fighters like you improve your skills and knowledge.")

    # settings for the LLM parameters to allow users to customize their coaching experience
        st.header("Coach Settings")
        temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, 0.1)
        top_k = st.slider("Number of Retrieved Documents", 3, 10, 6)
    
    st.markdown("---")
    st.info("Documents indexed via ingestion.py using local HuggingFace embeddings.")
    if st.button("🔄 Re-run Ingestion"):
        st.warning("Please run `python ingestion.py` in your terminal.")
    
    st.markdown("---")
    st.caption("This modular setup (ingestion → backend/core → Streamlit) keeps your AI MMA platform clean and scalable.")

# ================================== END OF SIDEBAR ============================================== #

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # List of (human, ai) tuples for memory

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ====================== CUSTOM CHAT INPUT ====================== #

# this forces the input box to clear after sending
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

user_input = st.text_input(
    label="Chat Input",
    placeholder="Hey there fighter! Hows your training going? Let's get to work! 🥊",
    key=f"user_input_{st.session_state.input_key}",
    label_visibility="collapsed"
)

# three buttons side by side for submitting text and voice input as well as microphone
col1, col2 = st.columns([3,2]) # Adjust column widths as needed
with col1:
    send_clicked = st.button("Send Message", type="primary", use_container_width=True)

with col2: 
    audio_input = st.audio_input("Record your question", key="audio_input")


# =========================== HANDLE LISTEN TO KNOX INPUT ===========================




# ====================== PROCESS SEND BUTTON ====================== #

if send_clicked and user_input and user_input.strip():
    prompt = user_input.strip()

    # Add user message to chat 
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response 
    with st.chat_message("assistant"):
        with st.spinner("Coach Knox is thinking... 🥊"):
            response = run_llm(
                query=prompt,
                chat_history=st.session_state.chat_history,
                temperature=temperature,
                k=top_k,
            )
            
            answer = response.get("answer", "Sorry, I couldn't generate a response.")
            sources = response.get("sources", [])
            images = response.get("images", [])

            # Save latest response for voice + images
            st.session_state.latest_answer = answer
            st.session_state.latest_images = images
            st.session_state.latest_sources = sources
            
            st.markdown(answer)

            # ====================== DISPLAY TECHNIQUE IMAGES ======================
            if images:
                st.markdown("### 📸 Relevant Technique Images:")
                cols = st.columns(3)
                for i, img_url in enumerate(images):
                    with cols[i % 3]:
                        try:
                            st.image(img_url, width="stretch")
                            st.caption(f"Technique Photo {i+1}")
                        except Exception as e:
                            st.warning(f"Couldn't load image: {img_url}")

            # ====================== SOURCES ======================
            if sources:
                with st.expander("📚 Sources from your RAG knowledge base"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**{i}.** [{source}]({source})")

            
            # Auto-play voice 
            try: 
                client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                audio_stream = client.text_to_speech.convert(
                    text=answer,
                    voice_id="pNInz6obpgDQGcFmaJgB",
                    model_id="eleven_turbo_v2_5",
                    output_format="mp3_44100_128"
                )
                audio_bytes = b"".join(chunk for chunk in audio_stream if chunk)
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            except Exception as e: 
                st.warning("Voice playback failed, but text is shown")


    # ====================== SAVE TO HISTORY (Only once) ======================
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append((prompt, answer))
    # Clear the input box after sending
    st.session_state.input_key += 1
    st.rerun()




# =========================== HANDLE MICROPHONE INPUT =============================== 

if audio_input is not None: 
    if "last_processed_audio" not in st.session_state or st.session_state.last_processed_audio != id(audio_input):

        st.session_state.last_processed_audio = id(audio_input) # Mark this audio as processed 


        #with st.spinner("Listening to your question..."):
        try: 
            recognizer = sr.Recognizer()
            #save the audio to a temp file 
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_input.getvalue())

            with sr.AudioFile("temp_audio.wav") as source:
                audio_data = recognizer.record(source)
            spoken_text = recognizer.recognize_google(audio_data, language="en-US")

            if spoken_text and spoken_text.strip():
                # removed because I don't need the input on the UI twice with a success msg
                #st.success(f"You said: {spoken_text}")

                # Use the spoken text as the prompt 
                prompt = spoken_text.strip()

                # Now process it the same way as typed input 
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get response from Knox 
                with st.chat_message("assistant"):
                    with st.spinner("Knox is thinking"):
                        response = run_llm(
                            query=prompt,
                            chat_history=st.session_state.chat_history, 
                            temperature=temperature,
                            k=top_k,
                        )

                        answer = response.get("answer", "Sorry, I couldn't generate a response.")
                        sources = response.get("sources", [])
                        images = response.get("images", [])
                        st.markdown(answer)
                        st.session_state.latest_answer = answer 

                        # Show Images 
                        if images: 
                            st.markdown(" Relevent Technique Images:")
                            cols = st.columns(3)
                            for i, img_url in enumerate(images):
                                with cols:
                                    st.image(img_url, use_column_width=True)
                                    st.caption(f"Technique {i+1}")

                        # Show Sources 
                        if sources: 
                            with st.expander("Sources for your RAG knowledge base"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"{i} ({source})")
                        



                    # Play voice response
                        try: 
                            client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                            audio_gen = client.text_to_speech.convert(
                                text=answer,
                                voice_id="pNInz6obpgDQGcFmaJgB",
                                model_id="eleven_turbo_v2_5",
                                output_format="mp3_44100_128",
                            )
                            audio_bytes = b"".join(chunk for chunk in audio_gen if chunk)
                            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                        except Exception as e: 
                            st.warning("Text response shown, but voice playback failed.")

               # Save to history (MUST be outside of the chat_message block)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.chat_history.append((prompt, answer))

            #Clear text input and rerun 
            #st.session_state.input_key += 1
            #st.rerun()

        except Exception as e:
            st.error(f"Could not understand audio: {str(e)}") 

    st.session_state.last_processed_audio = None 
