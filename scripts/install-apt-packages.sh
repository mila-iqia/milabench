#!/bin/bash

set -ex

SCRIPT_PATH=$(dirname "$0")

sudo apt-get update
sudo apt-get install -y $(cat ${SCRIPT_PATH}/apt_packages)
