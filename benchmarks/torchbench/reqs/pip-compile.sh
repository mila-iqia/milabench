#!/bin/bash
set -o errexit -o pipefail

# Script to partially-automate the compilation of pip requirements of models
# Expected to be executed in the bench code directory after `milabench install`

if [[ ! $(python3 -m pip freeze | grep "pip-tools") ]]
then
		python3 -m pip install pip -U
		python3 -m pip install pip-tools
fi

_REQS=reqs
_TB_ROOT=.
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
		--) break ;;
		-h | --help | *)
		if [[ "${_arg}" != "-h" ]] && [[ "${_arg}" != "--help" ]]
		then
			>&2 echo "Unknown option [${_arg}]"
		fi
		>&2 echo "Options for $(basename ${BASH_SOURCE[0]}) are:"
		>&2 echo "--reqs reqs dir"
		>&2 echo "--tb-root torchbench project root dir"
		exit 1
		;;
	esac
done

_MILABENCH_TBPATH=$(realpath "${_TB_ROOT}/torchbenchmark") \
python3 -m piptools compile \
	"$@" \
	"${_TB_ROOT}"/requirements.txt \
$(grep -o "model:.*" benchtest.yaml | cut -d" " -f2- | while read m
do
	# If the bench already has it's own setup.py, use it. Else, use the fake one
	setup_file=$([[ -f "${_TB_ROOT}/torchbenchmark/models/$m/setup.py" ]] &&
		echo "${_TB_ROOT}/torchbenchmark/models/$m/setup.py" ||
		echo "${_REQS}/$m/setup.py")
	# If the bench has a requirements.in and/or requirements-headless.in, use
	# them
	echo -n "$([[ -f "${_REQS}/$m/requirements.in" ]] &&
			echo "${_REQS}/$m/requirements.in" || echo "")" \
		"$([[ -f "${_REQS}/$m/requirements-headless.in" ]] &&
			echo "${_REQS}/$m/requirements-headless.in" || echo "")" \
		"$setup_file" ""
done)
