import os
import gradio as gr
import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Domyślny model (możesz nadpisać zmienną środowiskową ASR_MODEL_NAME)
MODEL_NAME = os.getenv("ASR_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")

# Bufory modeli dla różnych urządzeń
asr_model_mps = None
asr_model_cpu = None
asr_model_cuda = None


def transcribe_audio(audio_file_path: str):
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "Błąd: Nie wybrano pliku audio lub plik nie istnieje."

    try:
        audio = AudioSegment.from_file(audio_file_path)
    except FileNotFoundError:
        return "Błąd: Plik audio nie istnieje."
    except CouldntDecodeError:
        return "Błąd: Nie można odczytać pliku audio (sprawdź format/ffmpeg)."
    except Exception as e:
        return f"Błąd podczas wczytywania audio: {e}"

    length_minutes = len(audio) / (1000 * 60)
    print(f"Długość pliku: {length_minutes:.2f} minut.")

    # Wybór urządzenia: MPS dla krótszych plików, w innym wypadku CUDA/CPU
    device = "cpu"
    if torch.backends.mps.is_available() and length_minutes <= 8:
        device = "mps"
        print("Plik ≤ 8 min, używam MPS.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Znaleziono CUDA, używam GPU.")
    else:
        print("Brak GPU/MPS lub plik zbyt długi dla MPS, używam CPU.")
    print(f"Używane urządzenie: {device}")

    # Lazy-load model na odpowiednim urządzeniu i buforowanie między wywołaniami
    global asr_model_mps, asr_model_cpu, asr_model_cuda
    if device == "mps":
        if asr_model_mps is None:
            print(f"Ładowanie modelu {MODEL_NAME} na MPS...")
            asr_model_mps = (
                nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME).to(device)
            )
        asr_model = asr_model_mps
    elif device == "cuda":
        if asr_model_cuda is None:
            print(f"Ładowanie modelu {MODEL_NAME} na CUDA...")
            asr_model_cuda = (
                nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME).to(device)
            )
        asr_model = asr_model_cuda
    else:
        if asr_model_cpu is None:
            print(f"Ładowanie modelu {MODEL_NAME} na CPU...")
            asr_model_cpu = (
                nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME).to(device)
            )
        asr_model = asr_model_cpu

    print("Model załadowany. Transkrybuję...")
    try:
        with torch.inference_mode():
            transcriptions = asr_model.transcribe([audio_file_path])
        return transcriptions[0]
    except Exception as e:
        return f"Wystąpił błąd podczas transkrypcji: {e}"


iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Wybierz plik audio"),
    outputs=gr.Textbox(lines=10, label="Wynik transkrypcji"),
    title="Transkrypcja mowy na tekst",
    description="Wybierz plik audio, a model NVIDIA Parakeet wykona transkrypcję.",
)


if __name__ == "__main__":
    # (opcjonalnie na Mac/MPS) export PYTORCH_ENABLE_MPS_FALLBACK=1
    iface.launch()

