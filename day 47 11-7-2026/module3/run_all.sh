#!/bin/bash
# Mot entry point duy nhat. Mac dinh benchmark 30 frame cho moi runtime.
set -e
cd "$(dirname "$0")"
exec bash run_all_modern.sh "${1:-30}"
