def instrument_probe(ov):
    ov.probe(
        "/torchbenchmark.util.framework.vision.model_factory/TorchVisionModel/train(data as batch) > #endloop_data as step"
    ).give()

    # (
    #     ov.probe("/torchbenchmark.models.hf_Bert/Model/train(self) > #endloop__ as step")
    #     .augment(batch=lambda self: self.example_inputs["input_ids"])
    #     .give()
    # )
