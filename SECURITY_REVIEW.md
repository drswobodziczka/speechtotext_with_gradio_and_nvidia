# Security Review — simple_speech_to_text_with_parakeet

Data: 2025-09-27

## Zakres
Lokalna aplikacja web (Gradio) do transkrypcji audio → tekst przy użyciu NVIDIA NeMo (Parakeet). Uruchamiana na komputerze użytkownika, bez publikacji na zewnątrz.

## Werdykt
Bezpieczna do użytku lokalnego przy standardowych środkach ostrożności (zaufane pliki audio, aktualny ffmpeg i biblioteki). Aplikacja domyślnie nie wystawia się publicznie.

## Architektura i przepływ
- UI: Gradio nasłuchuje lokalnie (localhost) — brak `share=True` i niestawiamy `server_name` na wartość publiczną.
- Model: `nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)` pobiera wagi (pierwsze uruchomienie wymaga sieci).
- Audio: wczytywanie przez `pydub` (ffmpeg) i pomiar długości.
- Urządzenie: wybór między `MPS` (Apple Silicon dla ≤ 8 min), `CUDA` (jeśli dostępne) lub `CPU`.

## Potencjalne ryzyka
- Dekodowanie niezaufanego audio: `ffmpeg`/kodeki mogą mieć podatności. Zalecane użycie plików zaufanego pochodzenia i aktualnego ffmpeg.
- Łańcuch dostaw: instalacja paczek z PyPI oraz pobieranie modelu z zasobów NVIDIA. Warto rozważyć przypięcie wersji i regularne aktualizacje.
- Zużycie zasobów: długie pliki mogą obciążyć CPU/RAM, zwłaszcza na `CPU`.
- Ekspozycja w sieci lokalnej: pojawia się tylko, jeśli celowo ustawisz `share=True` lub `server_name='0.0.0.0'` — nie używamy tego w domyślnej konfiguracji.
- Obsługa ścieżek: aplikacja czyta jedynie plik wskazany przez uploader; brak zapisu/wykonywania zewnętrznych poleceń.

## Zalecenia i działania ograniczające ryzyko
- Utrzymuj aktualny: `ffmpeg`, `torch`, `nemo_toolkit` (regularne aktualizacje zabezpieczeń).
- Używaj wirtualnego środowiska (`venv`) i instaluj zależności przez `./scripts/install.sh`.
- Nie uruchamiaj z `share=True` ani z publicznym `server_name`; pozostaw localhost.
- Rozważ przypięcie wersji zależności w `requirements.txt` (dokładne wersje) i okresową rewizję.
- Opcjonalnie: dodanie limitu długości/rozmiaru w UI lub walidacji, by unikać przypadkowego wgrywania bardzo dużych plików.
- Apple Silicon: można ustawić `export PYTORCH_ENABLE_MPS_FALLBACK=1` by łagodniej radzić sobie z brakującymi operatorami.

## Uwagi operacyjne
- Pierwsze uruchomienie pobierze model (wymaga sieci). Kolejne uruchomienia mogą działać offline.
- Logi informują o wybranym urządzeniu (CPU/MPS/CUDA) i długości pliku — nie zawierają wrażliwych danych.

## Podsumowanie
Aplikacja nie otwiera połączeń sieciowych w runtime poza pobraniem modelu, nie uruchamia zewnętrznych poleceń poza `ffmpeg` (przez `pydub`) i pracuje lokalnie na plikach użytkownika. Przy zachowaniu powyższych zaleceń jest odpowiednia do lokalnego użytku.

