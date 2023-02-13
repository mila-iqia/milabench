#!/bin/bash
set -o errexit -o pipefail

# Script to partially-automate the compilation of pip requirements of models
# Expected to be executed in the bench code directory after `milabench install`

if [[ ! $(python3 -m pip freeze | grep "pip-tools") ]]
then
        python3 -m pip install pip -U
        python3 -m pip install pip-tools
fi

python3 -m piptools compile -v \
        --resolver=backtracking \
        --output-file requirements-bench.txt \
        reqs/requirements-bench.in \
        requirements.txt \
	$(grep -o "model:.*" benchtest.yaml | cut -d" " -f2- | while read m
do
	setup_file=$([[ -f "torchbenchmark/models/$m/setup.py" ]] && \
		echo "torchbenchmark/models/$m/setup.py" || \
		echo "reqs/$m/setup.py")
	echo -n "$([[ -f "reqs/$m/requirements.in" ]] && \
			echo "reqs/$m/requirements.in" || echo "")" \
		"$([[ -f "reqs/$m/requirements-headless.in" ]] && \
			echo "reqs/$m/requirements-headless.in" || echo "")" \
		"$setup_file" ""
done)
