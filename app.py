import os
import gradio as gr
import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Default model (override via ASR_MODEL_NAME)
MODEL_NAME = os.getenv("ASR_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")

# Model caches per device
asr_model_mps = None
asr_model_cpu = None
asr_model_cuda = None


def select_device(length_minutes: float) -> str:
    """Pick the best device based on availability and clip length."""
    if torch.backends.mps.is_available() and length_minutes <= 8:
        print("Clip <= 8 min, using MPS.")
        return "mps"
    if torch.cuda.is_available():
        print("CUDA available, using GPU.")
        return "cuda"
    print("No GPU/MPS or clip too long for MPS, using CPU.")
    return "cpu"


def get_asr_model(device: str):
    """Lazy-load and cache the ASR model on the requested device."""
    global asr_model_mps, asr_model_cpu, asr_model_cuda

    if device == "mps":
        if asr_model_mps is None:
            print(f"Loading model {MODEL_NAME} on MPS...")
            asr_model_mps = nemo_asr.models.ASRModel.from_pretrained(
                model_name=MODEL_NAME
            ).to(device)
        return asr_model_mps

    if device == "cuda":
        if asr_model_cuda is None:
            print(f"Loading model {MODEL_NAME} on CUDA...")
            asr_model_cuda = nemo_asr.models.ASRModel.from_pretrained(
                model_name=MODEL_NAME
            ).to(device)
        return asr_model_cuda

    # CPU
    if asr_model_cpu is None:
        print(f"Loading model {MODEL_NAME} on CPU...")
        asr_model_cpu = nemo_asr.models.ASRModel.from_pretrained(
            model_name=MODEL_NAME
        ).to(device)
    return asr_model_cpu


def transcribe_audio(audio_file_path: str):
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "Error: No audio file selected or file does not exist."

    try:
        audio = AudioSegment.from_file(audio_file_path)
    except FileNotFoundError:
        return "Error: Audio file does not exist."
    except CouldntDecodeError:
        return "Error: Could not decode audio (check format/ffmpeg)."
    except Exception as e:
        return f"Error while loading audio: {e}"

    length_minutes = len(audio) / (1000 * 60)
    print(f"Audio length: {length_minutes:.2f} minutes.")

    device = select_device(length_minutes)
    print(f"Selected device: {device}")

    asr_model = get_asr_model(device)

    print("Model loaded. Transcribing...")
    try:
        with torch.inference_mode():
            transcriptions = asr_model.transcribe([audio_file_path])
        return transcriptions[0]
    except Exception as e:
        return f"Transcription error: {e}"


iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Select audio file"),
    outputs=gr.Textbox(lines=10, label="Transcription"),
    title="Speech-to-Text Transcription",
    description="Upload an audio file and NVIDIA Parakeet will transcribe it.",
)


if __name__ == "__main__":
    # (optional on Mac/MPS) export PYTORCH_ENABLE_MPS_FALLBACK=1
    iface.launch()

