#!/bin/bash

SCRIPT_PATH=$(dirname "$0")

echo 'Install required apt packages'
$SCRIPT_PATH/install-apt-packages.sh
echo 'DONE: install required apt packages'

echo 'Install poetry'
$SCRIPT_PATH/install-poetry.sh
echo 'DONE: Install poetry'

echo 'Install milabench'
poetry install
echo 'DONE: Install milabench'
