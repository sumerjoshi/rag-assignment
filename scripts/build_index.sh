#!/bin/bash

set -euo pipefail

# run from the repo root no matter where this is called from
cd "$(dirname "$0")/.."

echo "Build the PDF vector store index"
echo ""
echo "This reads the PDFs in data/pdfs/, embeds them with the Fireworks"
echo "embedding model, and saves the index to storage/."
echo "It needs your Fireworks credentials in .env and may take a minute."
echo ""

read -r -p "Do you want to build the index now? [y/N] " answer
case "$answer" in
    [yY]|[yY][eE][sS]) ;;
    *)
        echo "Skipping index build."
        exit 0
        ;;
esac

# if an index is already there, ask before overwriting it
force_flag=""
if [ -d "storage" ]; then
    echo ""
    read -r -p "An index already exists at storage/. Rebuild from scratch? [y/N] " rebuild
    case "$rebuild" in
        [yY]|[yY][eE][sS]) force_flag="--force" ;;
        *) echo "Keeping the existing index." ;;
    esac
fi

# activate the venv if it exists
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo ""
echo "Running build_index..."
python -m src.ingest.build_index $force_flag

echo ""
echo "Done."
