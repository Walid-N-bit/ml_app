#!/bin/bash

MARKER="$HOME/.python_is_installed"

if [[ -f "$MARKER" ]]; then
  exit 0
fi

sudo apt update
sudo apt upgrade -y
sudo apt install python
sudo apt install python3-pip
clear

touch "$MARKER"
