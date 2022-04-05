import os


_MODELS_FACTORIES = ["HuggingFaceModel", "TimmModel", "TorchVisionModel"]


def instrument_probe(ov):
    loss = None

    (
        ov.probe("/torchbenchmark.util.framework.huggingface.model_factory/HuggingFaceModel/train(self) > #endloop__ as step")
        .augment(batch=lambda self: self.example_inputs["input_ids"])
        .augment(batch_size=lambda batch: len(batch))
        .give()
    )
    (
        ov.probe("/torchbenchmark.util.framework.huggingface.model_factory/HuggingFaceModel/train > loss")
        .take_last(1)["loss"]
        .map(float)
        .give("loss")
    )
    (
        ov.probe("/torchbenchmark.util.framework.timm.model_factory/TimmModel/train(self) > #endloop__ as step")
        .augment(batch=lambda self: self.example_inputs)
        .augment(batch_size=lambda batch: len(batch))
        .give()
    )
    @ov.probe("/torchbenchmark.util.framework.timm.model_factory/TimmModel/__init__(self) > #exit").ksubscribe
    def reg_timm_loss(self):
        (
            ov.probe("self.cfg.loss.__call__() as loss")
            .take_last(1)["loss"]
            .map(float)
            .give("loss")
        )
    (
        ov.probe("/torchbenchmark.util.framework.vision.model_factory/TorchVisionModel/train(real_input) > #endloop__ as step")
        .augment(batch=lambda real_input: real_input)
        .augment(batch_size=lambda batch: len(batch))
        .give()
    )
    # loss probed in ov.phases.load_script

    yield ov.phases.parse_args

    model = ov.options.ARGV[0]
    with open(f"{os.path.dirname(__file__)}/torchbenchmark/models/{model}/__init__.py", "r") as f:
        f = f.read()
        for mf in _MODELS_FACTORIES:
            if mf in f:
                break
        else:
            if model == "Background_Matting":
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train(data as batch) > #endloop_data as step")
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > lossG as loss")
                )
            elif model == "LearningToPaint":
                pass
            elif model == "dcgan":
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train(benchmark_pic as batch) > #endloop_i as step")
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > errD as loss")
                )
            elif model == "detectron2_maskrcnn":
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train(data as batch) > #endloop_data as step")
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > loss")
                )
            elif model in {"drq", "soft_actor_critic"}:
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > #endloop_step as step")
                    .give()
                )
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > reward")
                    .slice(-1)["reward"]
                    .map(float)
                    .give("reward")
                )
            elif model == "fastNLP_Bert":
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train(batch_x as batch) > #endloop_batch_x as step")
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > loss")
                )
            elif model in {"maml", "maml_omniglot"}:
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train(self) > #endloop__ as step")
                    .augment(batch=lambda self: self.example_inputs[0])
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
            elif model in {"mobilenet_v2_quantized_qat", "resnet50_quantized_qat"}:
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train(self) > #endloop__ as step")
                    .augment(batch=lambda self: self.example_inputs[0])
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                @ov.probe(f"/torchbenchmark.models.{model}/Model/train > loss").ksubscribe
                def reg(loss):
                    (
                        ov.probe("loss.__call__() as loss")
                        .last()["loss"]
                        .map(float)
                        .give("loss")
                    )
            elif model == "moco":
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train(self) > #endloop_i as step")
                    .augment(batch=lambda self: self.example_inputs)
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > loss")
                )
            elif model == "nvidia_deeprecommender":
                pass
            elif model == "opacus_cifar10":
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > images as batch")
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > loss")
                )
            elif model.startswith("pyhpc_"):
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/eval(example_inputs) > #endloop_i as step")
                    .augment(batch=lambda example_inputs: example_inputs[0])
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
            elif model == "pytorch_CycleGAN_and_pix2pix":
                pass
            elif model == "pytorch_stargan":
                pass
            elif model == "speech_transformer":
                (
                    ov.probe(f"/torchbenchmark.models.{model}.config/SpeechTransformerTrainConfig/_run_one_epoch(padded_input as batch) > #endloop_i as step")
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}.config/SpeechTransformerTrainConfig/_run_one_epoch > loss")
                )
            elif model == "timm_efficientdet":
                (
                    ov.probe(f"/torchbenchmark.models.{model}.train/train_epoch(input as batch) > #endloop_batch_idx as step")
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}.train/train_epoch > loss")
                )
            elif model == "tts_angular":
                (
                    ov.probe(f"/torchbenchmark.models.{model}.angular_tts_main/TTSModel/_train(data as batch) > #endloop__ as step")
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}.angular_tts_main/TTSModel/_train > loss")
                )
            elif model == "vision_maskrcnn":
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train(images as batch) > #endloop__ as step")
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > losses as loss")
                )
            elif model == "yolov3":
                pass
            else:
                (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train(self) > #endloop__ as step")
                    .augment(batch=lambda self: self.example_inputs[0])
                    .augment(batch_size=lambda batch: len(batch))
                    .give()
                )
                loss = (
                    ov.probe(f"/torchbenchmark.models.{model}/Model/train > loss")
                )

    if loss is not None:
        loss.last()["loss"].map(float).give("loss")

    yield ov.phases.load_script

    # loss of TorchVisionModel
    @ov.probe("//run_one_step > func").kmap(model=lambda func: func.__self__).ksubscribe
    def reg_vision_loss(model):
        if hasattr(model, "loss_fn"):
            (
                ov.probe("model.loss_fn.__call__() as loss")
                .last()["loss"]
                .map(float)
                .give("loss")
            )

    ov.probe("//run_one_step > gpu_time").give()
    ov.probe("//run_one_step > cpu_dispatch_time").give()
    ov.probe("//run_one_step > cpu_walltime").give()
    ov.probe("//run_one_step > tflops").give()
    ov.probe("//run_one_step > func").kmap(model=lambda func: func.__self__).give()
