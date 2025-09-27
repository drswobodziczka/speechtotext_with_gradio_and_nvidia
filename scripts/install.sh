#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Błąd: Aktywuj najpierw wirtualne środowisko (source .venv/bin/activate)." >&2
  exit 1
fi

python -m pip install -U pip
python -m pip install -r requirements.txt

echo "Instalacja zakończona w venv: $VIRTUAL_ENV"

