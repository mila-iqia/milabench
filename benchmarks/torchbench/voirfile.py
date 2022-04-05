def instrument_probe(ov):
    # ov.probe(
    #     "/torchbenchmark.util.framework.vision.model_factory/TorchVisionModel/train(data as batch) > #endloop_data as step"
    # ).give()

    (
        ov.probe(
            "/torchbenchmark.util.framework.huggingface.model_factory/HuggingFaceModel/train(self) > #endloop__ as step"
        )
        .augment(batch=lambda self: self.example_inputs["input_ids"])
        .augment(batch_size=lambda batch: len(batch))
        .give()
    )
