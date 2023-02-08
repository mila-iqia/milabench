#!/bin/bash
set -o errexit -o pipefail

# Script to partially-automate the compilation of pip requirements of models
# Expected to be executed in the bench code directory after `milabench install`

if [[ ! $(python3 -m pip freeze | grep "pip-tools") ]]
then
        python3 -m pip install pip -U
        python3 -m pip install pip-tools
fi

models=(
	BERT_pytorch
	Background_Matting
	LearningToPaint
	Super_SloMo
	alexnet
	attention_is_all_you_need_pytorch
	dcgan
	demucs
	densenet121
	detectron2_maskrcnn
	dlrm
	drq
	fastNLP_Bert
	hf_Albert
	hf_Bart
	hf_Bert
	hf_BigBird
	hf_DistilBert
	hf_GPT2
	hf_Longformer
	hf_Reformer
	hf_T5
	maml
	maml_omniglot
	mnasnet1_0
	mobilenet_v2
	mobilenet_v2_quantized_qat
	mobilenet_v3_large
	moco
	nvidia_deeprecommender
	opacus_cifar10
	pyhpc_equation_of_state
	pyhpc_isoneutral_mixing
	pyhpc_turbulent_kinetic_energy
	pytorch_CycleGAN_and_pix2pix
	pytorch_stargan
	pytorch_struct
	pytorch_unet
	resnet18
	resnet50
	resnet50_quantized_qat
	resnext50_32x4d
	shufflenet_v2_x1_0
	soft_actor_critic
	speech_transformer
	squeezenet1_1
	tacotron2
	timm_efficientdet
	timm_efficientnet
	timm_nfnet
	timm_regnet
	timm_resnest
	timm_vision_transformer
	timm_vovnet
	tts_angular
	vgg16
	vision_maskrcnn
	yolov3)

for m in "${models[@]}"
do
	setup_file=$([[ -f "torchbenchmark/models/$m/setup.py" ]] && echo "torchbenchmark/models/$m/setup.py" || echo "setup.py")
	_MB_MODEL="$m" python3 -m piptools compile -v \
		--resolver=backtracking \
		--output-file requirements-"$m".txt \
		requirements-bench.in \
		$([[ -f "requirements-$m.in" ]] && echo "requirements-$m.in" || echo "") \
		requirements.txt \
		"$setup_file"

	if [[ -x "requirements-$m-headless.in" ]]
	then
		_MB_MODEL="$m" python3 -m piptools compile -v \
			--resolver=backtracking \
			--output-file requirements-"$m"-headless.txt \
			requirements-bench.in \
			requirements-"$m"-headless.in \
			requirements.txt \
			"$setup_file"
	fi
done
