#!/bin/bash

for gpu in a100 a100l l40s rtx8000 v100 h100
do
# 	squeue -u$USER -o "%A %j" | grep " mb-scale-${gpu}-1$" || \
# 		sbatch --gpus-per-task=${gpu}:1 --cpus-per-task=6 --job-name "mb-scale-${gpu}-1" scale.sh "${gpu}"
	squeue -u$USER -o "%A %j" | grep "mb-scale-${gpu}-4$" || \
		sbatch --gpus-per-task=${gpu}:4 --job-name "mb-scale-${gpu}-4" scale.sh "${gpu}"
done

gpu=h100
# while true
# do
# 	squeue -u$USER -o "%j" | grep "^mb-scale-${gpu}-1$" && sleep 5m || \
# 		{ sbatch --wait --time=3:0:0 --partition short-unkillable --gpus-per-task=${gpu}:1 --cpus-per-task=6 --job-name "mb-scale-${gpu}-1" scale.sh "${gpu}" \
# 			&& break ; }
# done

while true
do
	squeue -u$USER -h -n "mb-scale-${gpu}-4" -o"%L" | grep "" >/dev/null || \
		sbatch --time=3:0:0 --partition short-unkillable --gpus-per-task=${gpu}:4 --job-name "mb-scale-${gpu}-4" scale.sh "${gpu}"

	{ date ; squeue -u$USER -n "mb-scale-${gpu}-4" ; } | timeout --foreground 5m less && break

	time_left=$(squeue -u$USER -h -n "mb-scale-${gpu}-4" -o"%L")
	d=$(echo $time_left | cut -d"-" -f1)
	hms=$(echo $time_left | cut -d"-" -f2)

	# echo $time_left
	# echo $d
	# echo $hms

	if [[ "$d" != "$hms" ]]
	then
		# some days left
		continue
	fi

	h=$(echo $hms | cut -d":" -f1)
	m=$(echo $hms | cut -d":" -f2)
	s=$(echo $hms | cut -d":" -f3)

	# echo $h
	# echo $m
	# echo $s

	if [[ -z "$s" ]]
	then
		s=$m
		m=$h
		h=
	fi

	# echo $h
	# echo $m
	# echo $s

	if [[ -z "$h" ]] && [[ "$m" -lt "15" ]]
	then
		scontrol requeue $(squeue -u$USER -h -n "mb-scale-${gpu}-4" -o"%A")
	fi
done
