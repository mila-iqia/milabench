#!/bin/bash

set -ex

SUDO=""

if [[ "$EUID" -ne 0 ]]; then
    SUDO="sudo"
fi

SCRIPT_PATH=$(dirname "$0")

$SUDO apt-get update
$SUDO apt-get install -y $(cat ${SCRIPT_PATH}/apt_packages)
