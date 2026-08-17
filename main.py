import base64
import os
import wave
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from streamlit.errors import StreamlitSecretNotFoundError

# This is to load environment
load_dotenv()


def get_setting(name, default=None):
    """Read configuration from the environment or Streamlit secrets."""
    if value := os.getenv(name):
        return value
    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


GOOGLE_API_KEY = get_setting("GOOGLE_API_KEY")
GEMINI_STT_MODEL = get_setting("GEMINI_STT_MODEL", "gemini-3.6-flash")
GEMINI_TTS_MODEL = get_setting(
    "GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"
)
GEMINI_TTS_VOICE = get_setting("GEMINI_TTS_VOICE", "Kore")


def get_pdf_text(documents):
    """Extract text while isolating failures to individual files or pages."""
    pages = []
    issues = []

    for document in documents:
        filename = getattr(document, "name", "uploaded PDF")
        try:
            if hasattr(document, "seek"):
                document.seek(0)
            pdf_reader = PdfReader(document, strict=False)

            for page_number, page in enumerate(pdf_reader.pages, start=1):
                try:
                    if page_text := page.extract_text():
                        pages.append(page_text)
                except Exception as error:
                    issues.append(f"{filename}, page {page_number}: {error}")
        except Exception as error:
            issues.append(f"{filename}: {error}")

    if not pages:
        detail = f" First extraction error: {issues[0]}" if issues else ""
        raise ValueError(
            "No readable text was found. The PDF may be scanned, encrypted, or damaged."
            + detail
        )

    return "\n".join(pages), issues


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10_000,
        chunk_overlap=1_000,
    )
    return splitter.split_text(text)


def create_embeddings():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is missing.")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )


def build_pdf_index(chunks):
    """Create an in-memory vector index for the current user session."""
    if not chunks:
        raise ValueError("No readable text was found in the uploaded PDF files.")

    return FAISS.from_texts(texts=chunks, embedding=create_embeddings())


def index_is_ready():
    return st.session_state.get("vector_store") is not None


def get_answer_prompt():
    return PromptTemplate(
        template="""
Answer the question using only the provided context.
Give a clear, direct answer that is suitable for reading aloud.

If the answer is not available in the context, say exactly:
"Answer to the question is not available in the provided document."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )


@st.cache_resource(show_spinner=False)
def get_answer_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GOOGLE_API_KEY,
    )


def answer_pdf_question(question):
    """Retrieve relevant PDF passages and return the generated answer text."""
    if not index_is_ready():
        raise FileNotFoundError(
            "No PDF index is available. Upload PDF files and select 'Process PDFs' first."
        )

    vector_store = st.session_state.vector_store
    documents = vector_store.similarity_search(question)
    if not documents:
        return "No matching content was found in the uploaded PDF files."

    context = "\n\n".join(document.page_content for document in documents)
    prompt = get_answer_prompt().format(context=context, question=question)
    response = get_answer_model().invoke(prompt)
    if isinstance(response.content, str):
        return response.content.strip()

    text_blocks = (
        block.get("text", "")
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    answer = "".join(text_blocks).strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty answer.")
    return answer


@st.cache_resource(show_spinner=False)
def get_google_audio_client():
    if not GOOGLE_API_KEY:
        return None
    return genai.Client(api_key=GOOGLE_API_KEY)


def require_google_audio_client():
    client = get_google_audio_client()
    if client is None:
        raise RuntimeError("GOOGLE_API_KEY is missing.")
    return client


def transcribe_audio(audio_file):
    """Convert a Streamlit microphone recording into text."""
    audio_bytes = (
        audio_file.getvalue() if hasattr(audio_file, "getvalue") else audio_file.read()
    )
    content_type = getattr(audio_file, "type", "audio/wav")
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise ValueError("The recording is too large. Please record a shorter question.")

    transcript = require_google_audio_client().interactions.create(
        model=GEMINI_STT_MODEL,
        input=[
            {
                "type": "text",
                "text": (
                    "Transcribe the spoken question accurately. "
                    "Return only the words spoken, with no commentary."
                ),
            },
            {
                "type": "audio",
                "data": base64.b64encode(audio_bytes).decode("ascii"),
                "mime_type": content_type,
            },
        ],
    )
    return transcript.output_text.strip()


def pcm_to_wav(pcm_bytes):
    """Wrap Gemini's 24 kHz mono PCM output in a WAV container."""
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(pcm_bytes)
    return output.getvalue()


def generate_speech(text):
    """Generate a WAV voice reply and return it as bytes."""
    client = require_google_audio_client()
    prompt = f"Synthesize natural, clear speech. Read this transcript exactly:\n{text}"

    for attempt in range(2):
        try:
            response = client.interactions.create(
                model=GEMINI_TTS_MODEL,
                input=prompt,
                response_format={"type": "audio"},
                generation_config={
                    "speech_config": [{"voice": GEMINI_TTS_VOICE}]
                },
            )
            if response.output_audio is None:
                raise RuntimeError("The speech model did not return audio.")
            encoded_audio = response.output_audio.data
            pcm_bytes = (
                base64.b64decode(encoded_audio)
                if isinstance(encoded_audio, str)
                else encoded_audio
            )
            return pcm_to_wav(pcm_bytes)
        except ServerError:
            if attempt == 1:
                raise


def initialize_session_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pdf_processed", False)
    st.session_state.setdefault("vector_store", None)


def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("audio"):
                st.audio(message["audio"], format="audio/wav")


def process_submission(submission):
    typed_question = submission.text.strip()
    recording = submission.audio
    transcript = ""

    if recording is not None:
        with st.spinner("Transcribing your question..."):
            transcript = transcribe_audio(recording)

    question_parts = [part for part in (typed_question, transcript) if part]
    return "\n".join(question_parts), recording is not None


def render_sidebar():
    with st.sidebar:
        st.header(":material/picture_as_pdf: Upload PDFs")
        documents = st.file_uploader(
            "Choose one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button(
            "Process PDFs",
            icon=":material/document_scanner:",
            width="stretch",
        ):
            if not documents:
                st.error("Upload at least one PDF first.")
            else:
                st.session_state.pdf_processed = False
                st.session_state.vector_store = None
                st.session_state.messages = []
                try:
                    with st.spinner("Reading and indexing the PDF files..."):
                        text, extraction_issues = get_pdf_text(documents)
                        st.session_state.vector_store = build_pdf_index(
                            get_text_chunks(text)
                        )
                except Exception as error:
                    st.session_state.pdf_processed = False
                    st.session_state.vector_store = None
                    st.error(f"The PDF files could not be processed: {error}")
                else:
                    st.session_state.pdf_processed = True
                    if extraction_issues:
                        st.warning(
                            "PDFs processed, but "
                            f"{len(extraction_issues)} page(s) could not be read. "
                            "Questions will use the successfully extracted pages."
                        )
                    else:
                        st.success(
                            "PDFs processed. You can now ask questions by text or voice."
                        )

        if st.session_state.pdf_processed and index_is_ready():
            st.success("A PDF index is ready.", icon=":material/check_circle:")
        else:
            st.info("Process PDF files to enable the chat.")


def main():
    st.set_page_config(
        page_title="Voice Chat with PDFs",
        page_icon=":material/record_voice_over:",
    )
    initialize_session_state()

    st.title("Chat with PDFs", anchor=False)
    st.caption(
        "Type a question or select the microphone in the message box. "
        "Voice replies are AI-generated."
    )

    if not GOOGLE_API_KEY:
        st.error("Add GOOGLE_API_KEY to your .env file, then restart the app.")
        st.stop()

    render_sidebar()
    render_chat_history()

    ready = st.session_state.pdf_processed and index_is_ready()
    submission = st.chat_input(
        "Ask a question about your PDFs",
        key="pdf_question",
        accept_audio=True,
        audio_sample_rate=16_000,
        disabled=not ready,
        submit_mode="disable",
    )

    if not submission:
        return

    try:
        question, used_microphone = process_submission(submission)
    except Exception as error:
        st.error(f"Your recording could not be transcribed: {error}")
        return

    if not question:
        st.warning("Type or record a question before sending it.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
        if used_microphone:
            st.caption("Transcribed from microphone")

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the PDFs..."):
                answer = answer_pdf_question(question)
        except Exception as error:
            st.error(f"The question could not be answered: {error}")
            return

        st.markdown(answer)

        audio_bytes = None
        try:
            with st.spinner("Creating the voice reply..."):
                audio_bytes = generate_speech(answer)
        except Exception as error:
            st.warning(f"The text answer is ready, but the voice reply failed: {error}")
        else:
            st.audio(audio_bytes, format="audio/wav", autoplay=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "audio": audio_bytes}
    )


if __name__ == "__main__":
    main()
