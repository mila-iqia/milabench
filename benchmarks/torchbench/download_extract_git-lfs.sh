#!/bin/bash
set -o errexit -o pipefail -o noclobber

function exit_on_error_code {
        local _ERR=$?
        while [[ $# -gt 0 ]]
        do
                local _arg="$1"; shift
                case "${_arg}" in
                        --err)
                        if [[ ${_ERR} -eq 0 ]]
                        then
                                _ERR=$1
                        fi
                        shift
                        ;;
                        -h | --help)
                        if [[ "${_arg}" != "-h" ]] && [[ "${_arg}" != "--help" ]]
                        then
                                >&2 echo "Unknown option [${_arg}]"
                        fi
                        >&2 echo "Options for ${FUNCNAME[0]} are:"
                        >&2 echo "[--err INT] use this exit code if '\$?' is 0 (optional)"
                        >&2 echo "ERROR_MESSAGE error message to print"
                        exit 1
                        ;;
                        *) set -- "${_arg}" "$@"; break ;;
                esac
        done

        if [[ ${_ERR} -ne 0 ]]
        then
                >&2 echo "$(tput setaf 1)ERROR$(tput sgr0): $1: ${_ERR}"
                exit ${_ERR}
        fi
}

_VERSION=v2.13.3

mkdir -p bin/git-lfs-linux-amd64-${_VERSION}

rm -f bin/sha256sums
echo "03197488f7be54cfc7b693f0ed6c75ac155f5aaa835508c64d68ec8f308b04c1  git-lfs-linux-amd64-${_VERSION}.tar.gz" > bin/sha256sums

wget https://github.com/git-lfs/git-lfs/releases/download/${_VERSION}/git-lfs-linux-amd64-${_VERSION}.tar.gz -O bin/git-lfs-linux-amd64-${_VERSION}.tar.gz
pushd bin/ >/dev/null
sha256sum -c sha256sums
popd >/dev/null
exit_on_error_code "Failed to download git-lfs"

tar -C bin/git-lfs-linux-amd64-${_VERSION} -xf bin/git-lfs-linux-amd64-${_VERSION}.tar.gz
exit_on_error_code "Failed to extract git-lfs"


pushd bin/ >/dev/null
ln -sf git-lfs-linux-amd64-${_VERSION}/git-lfs .
popd >/dev/null
