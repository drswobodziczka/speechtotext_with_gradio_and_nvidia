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

## Warm-up (opcjonalnie)

- Pre‑pull modelu do cache (szybszy start UI):
```
python - <<'PY'
import nemo.collections.asr as asr
asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')
print('Model w cache')
PY
```

- Zbudowanie cache czcionek Matplotlib (jednorazowo):
```
python -c "from matplotlib import font_manager as fm; _=fm.FontManager(); print('Matplotlib font cache ready')"
```

## Szybsze pobieranie (HF Transfer)

- W niektórych środowiskach przyspiesza pobieranie modeli:
```
python -m pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
# (opcjonalnie) huggingface-cli login
```

## Mniejszy model do szybkich testów

- Zamiast dużego Parakeet ustaw tymczasowo mniejszy ASR, np.:
```
export ASR_MODEL_NAME="stt_en_fastconformer_ctc_small"
python app.py
```
Po testach wróć do domyślnego Parakeet.

## Hypotheses — wyjaśnienie

- Czym jest hipoteza: pojedynczy kandydat transkrypcji wygenerowany przez model ASR.
- Co zwraca NeMo:
  - Bez opcji: `transcribe(...) -> List[str]` (sam tekst najlepszej hipotezy).
  - Z opcją: `transcribe(..., return_hypotheses=True) -> List[Hypothesis]`.
- Pola w `Hypothesis` (najważniejsze):
  - `text` — zdekodowany tekst (to zwykle chcesz pokazać użytkownikowi).
  - `score` — wynik w skali log‑prawdopodobieństwa (zwykle ujemny). Bliżej 0 = „lepszy” w porównaniu do innych hipotez dla TEGO samego nagrania i modelu.
  - `y_sequence` — sekwencja ID tokenów (wewnętrzna reprezentacja przed dekodowaniem; stąd „tablica numerków”).

Jak interpretować `score`:
- Nie jest to „% pewności”. To suma/miara log‑prawdopodobieństw, więc dłuższe klipy często mają bardziej ujemne wartości.
- Porównuj tylko hipotezy wygenerowane dla TEGO SAMEGO nagrania i modelu (ranking). Nie porównuj wartości między różnymi klipami czy modelami.
- Jeśli chcesz przybliżone porównanie między klipami, możesz policzyć „score na token” (np. `score / len(y_sequence)`), ale traktuj to orientacyjnie.

Po co hipotezy (n‑best):
- Debugowanie i analiza jakości (czy druga/trzecia propozycja nie jest lepsza?).
- Reguły post‑processingowe albo dodatkowe modele (np. language model) mogą „przeważać” między kilkoma kandydatami.

Krótkie przykłady
```
# 1) Najlepsza hipoteza (tekst + wynik)
python - <<'PY'
import nemo.collections.asr as asr
m = asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')
h = m.transcribe(['sample.wav'], return_hypotheses=True)[0]
print('TEXT:', h.text)
print('SCORE (logP):', h.score)
PY

# 2) Top-3 hipotezy (jeśli dekoder je zwraca)
python - <<'PY'
import nemo.collections.asr as asr
m = asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')
hyps = m.transcribe(['sample.wav'], return_hypotheses=True)
for i, hyp in enumerate(hyps[:3], 1):
    print(f'#{i}', hyp.score, hyp.text)
PY

# 3) Zamiana ID tokenów na tekst (ciekawostka)
python - <<'PY'
import nemo.collections.asr as asr
m = asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')
h = m.transcribe(['sample.wav'], return_hypotheses=True)[0]
ids = h.y_sequence.tolist()
print(m.tokenizer.ids_to_text(ids))
PY
```

## Credits

- Based on: https://huggingface.co/spaces/areksmyk/speechtotext/tree/main
