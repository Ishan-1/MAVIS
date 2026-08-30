#!/bin/bash
# reset_chroma.sh — Wipe ChromaDB on-disk collections and recreate them fresh.
#
# Run this if you see a chromadb KeyError: '_type' crash on startup.
# This happens when the chromadb version changes between runs and
# the on-disk schema becomes incompatible.  No real memory data
# is lost during the prototype phase since data accumulates slowly.
#
# Usage:
#   ./scripts/reset_chroma.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "[reset_chroma] Wiping stale ChromaDB collections in ${ROOT_DIR}..."
find memories -type d -name "chroma" -exec rm -rf {} + 2>/dev/null || true
mkdir -p memories/interpreter/short_term/chroma memories/interpreter/long_term/chroma

echo "[reset_chroma] Recreating with current chromadb version..."
python3 -c "
import chromadb
st = chromadb.PersistentClient(path='memories/interpreter/short_term/chroma')
lt = chromadb.PersistentClient(path='memories/interpreter/long_term/chroma')
st.get_or_create_collection('interpreter_short_term', metadata={'hnsw:space': 'cosine'})
lt.get_or_create_collection('interpreter_long_term', metadata={'hnsw:space': 'cosine'})
print('  interpreter_short_term: OK')
print('  interpreter_long_term:  OK')
print('[reset_chroma] Done. Run: python main.py')
"
