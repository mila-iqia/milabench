#!/bin/bash
set -o errexit -o pipefail

# Script to partially-automate the compilation of pip requirements of models
# Expected to be executed in the bench code directory after `milabench install`

_REQS=reqs
_TB_ROOT=.
_CONFIG=benchtest.yaml
declare -a _MODELS=()
while [[ $# -gt 0 ]]
do
	_arg="$1"; shift
	case "${_arg}" in
		--reqs) _REQS="$1"; shift
		echo "reqs = [${_REQS}]"
		;;
		--tb-root) _TB_ROOT="$1"; shift
		echo "tb-root = [${_TB_ROOT}]"
		;;
		--config) _CONFIG="$1"; shift
		echo "config = [${_CONFIG}]"
		;;
		-m|--model) _MODELS+=("$1"); shift
		;;
		--) break ;;
		-h | --help | *)
		if [[ "${_arg}" != "-h" ]] && [[ "${_arg}" != "--help" ]]
		then
			>&2 echo "Unknown option [${_arg}]"
		fi
		>&2 echo "Options for $(basename ${BASH_SOURCE[0]}) are:"
		>&2 echo "--reqs DIR reqs dir"
		>&2 echo "--tb-root DIR torchbench project root dir"
		>&2 echo "--config FILE config.yaml"
		>&2 echo "-m | --model MODEL model name to compute requirements for. This option can be used multiple times. This option taks precedence over --config"
		exit 1
		;;
	esac
done

if (( ${#_MODELS[@]} == 0 ))
then
	while read m
	do
		_MODELS+=("$m")
	done < <(grep -o "model:.*" "${_CONFIG}" | cut -d" " -f2-)
fi

for m in "${_MODELS[@]}"
do
	# If the bench has a setup.py, use it instead of faking the install.py's
	# requirements
	if [[ ! -f "${_TB_ROOT}/torchbenchmark/models/$m/"setup.py ]]
	then
		_MILABENCH_TBPATH=$(realpath "${_TB_ROOT}/torchbenchmark") python3 \
			"${_REQS}/$m/"get_install_py_requirements.py \
			>"${_REQS}/$m/"requirements-install_py.in
		[[ -s "${_REQS}/$m/"requirements-install_py.in ]] || rm "${_REQS}/$m/"requirements-install_py.in
	fi
done

python3 -m piptools compile \
	"$@" \
	"${_TB_ROOT}"/requirements.txt \
$(for m in "${_MODELS[@]}"
do
	# If the bench already has it's own setup.py, use it. Else, use the fake one
	if [[ -f "${_TB_ROOT}/torchbenchmark/models/$m/setup.py" ]]
	then
		setup_or_install_req_file="${_TB_ROOT}/torchbenchmark/models/$m/"setup.py
	elif [[ -f "${_REQS}/$m/"requirements-install_py.in ]]
	then
		setup_or_install_req_file="${_REQS}/$m/"requirements-install_py.in
	fi
	# If the bench has a requirements.in and/or requirements-headless.in, use
	# them
	echo -n "$([[ -f "${_REQS}/$m/requirements.in" ]] &&
			echo "${_REQS}/$m/requirements.in" || echo "")" \
		"$([[ -f "${_REQS}/$m/requirements-headless.in" ]] &&
			echo "${_REQS}/$m/requirements-headless.in" || echo "")" \
		"$setup_or_install_req_file" ""
done)
