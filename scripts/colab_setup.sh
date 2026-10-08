#!/usr/bin/env bash
set -e
pip install -r requirements.txt
pip uninstall -y torchao || true
mkdir -p data/meld data/masac outputs
