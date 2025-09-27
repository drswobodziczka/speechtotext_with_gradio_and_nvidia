# simple_speech_to_text_with_parakeet

Jednookienkowy transkryptor audio→tekst (Gradio + NVIDIA NeMo Parakeet) z automatycznym wyborem urządzenia (CPU/MPS/CUDA).
„To prosty projekt do szybkiej transkrypcji audio bez konfiguracji serwera.”

## Jaki model pobieramy?

Domyślnie: nvidia/parakeet-tdt-0.6b-v3 (pobierany przez from_pretrained).
Zmiana modelu bez edycji kodu:

- `export ASR_MODEL_NAME="nvidia/parakeet-tdt-1.1b"` (przykład, jeśli dostępny)

## Zależności — po co są?

- gradio — web‑UI do wgrywania audio i podglądu wyniku.
- torch — tensory i urządzenia (CPU/CUDA/MPS).
- nemo_toolkit[asr] — modele ASR (Parakeet), API ASRModel.transcribe.
- pydub — wczytywanie audio i pomiar długości (wymaga ffmpeg).
- onnx — format modeli (eksport/kompatybilność).
- onnxruntime — uruchamianie modeli ONNX (opcjonalne, nieużywane bezpośrednio).
- ml_dtypes — dodatkowe typy numeryczne (kompatybilność z NumPy 2.x).

## Wymagania

- Python 3.10+
- ffmpeg (dla pydub), np. macOS: `brew install ffmpeg`

## Instalacja (tylko wewnątrz aktywnego venv)

```
python3 -m venv .venv
source .venv/bin/activate
./scripts/install.sh
```

## Uruchomienie

```
python app.py
```

Otwórz link Gradio z terminala.

## Notatki

- `ASRModel.transcribe()` zwraca listę stringów; po hipotezy użyj: `transcribe(..., return_hypotheses=True)` i `transcriptions[0].text`.
- (Apple Silicon) można ustawić: `export PYTORCH_ENABLE_MPS_FALLBACK=1`.

