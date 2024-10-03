function _get_options {
	set +o | cut -d' ' -f2- | while read set_option
	do
		echo "${set_option}"
	done
}

function _set_options {
	while [[ $# -gt 0 ]]
	do
		local _arg="$1"; shift
		case "${_arg}" in
			-o) set -o "$1"; shift ;;
			+o) set +o "$1"; shift ;;
			-h | --help | *)
			exit 1
			;;
		esac
	done
}

_options=$(_get_options)
set -o errexit -o pipefail

_NAME="{{milabench_env}}"
conda --version >&2 2>/dev/null || module load anaconda/3
conda activate ${_NAME} || conda create -y -n ${_NAME} "python={{python_version}}" virtualenv
conda activate ${_NAME}

_set_options $_options
unset _options