#!/bin/bash
set -e

echo "🦙 Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

echo "⏳ Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/ > /dev/null 2>&1; do
    sleep 1
done
echo "✅ Ollama ready"

echo "🚀 Starting OBSIDIAN Neural Provider..."
exec python3 provider.py

kill $OLLAMA_PID