#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve LLAMA3 model..."
ollama pull llama3.1:8b
echo "🟢 LLAMA3 model Done!"

# echo "🔴 Retrieve Qwen2.5 model..."
# ollama pull qwen2.5:0.5b
# echo "🟢 Qwen2.5 model Done!"

echo "🔴 Retrieve nomic-embed-text model..."
ollama pull nomic-embed-text
echo "🟢 nomic-embed-text model Done!"

# Wait for Ollama process to finish.
wait $pid