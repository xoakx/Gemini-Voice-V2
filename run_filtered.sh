#!/bin/bash
# Gemini Voice Wrapper v5.1: Instant Stream Filtering
cd /home/kms/gemini-voice-v2

# 1. Capture stdout/stderr
# 2. Use sed to strip DEBUG lines and ANSI escape codes instantly
# 3. Ensure unbuffered output for real-time Cockpit updates
.venv/bin/python3 -u src/modules/pro.py 2>&1 | stdbuf -oL -eL sed -u '/DEBUG/d; s/\x1b\[[0-9;]*m//g'
