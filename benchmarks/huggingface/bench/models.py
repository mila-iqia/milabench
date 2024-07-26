from types import SimpleNamespace as NS

import transformers
from transformers import AutoConfig, BertConfig, BigBirdConfig, ReformerConfig

models = {}


def register_model(fn):
    models[fn.__name__] = fn
    return fn


def _make(category, config):
    return getattr(transformers, category).from_config(config)



def synthetic_dataset(info, args):
    from .synth import SyntheticData, generators
    from torch.utils.data import DataLoader

    data = SyntheticData(
        n=args.batch_size,
        repeat=100000,
        generators=generators[info.category](info),
    )
    return DataLoader(
        data, batch_size=args.batch_size, num_workers=args.num_workers
    )


@register_model
def Opt350m(args):
    category = "AutoModelForCausalLM"
    config = AutoConfig.from_pretrained("facebook/opt-350m")
    return NS(
        category=category,
        config=config,
        train_length=config.max_position_embeddings,
        eval_length=config.max_position_embeddings,
        model=_make(category, config),
    )


@register_model
def GPT2(args):
    category = "AutoModelForCausalLM"
    config = AutoConfig.from_pretrained("gpt2")
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=1024,
        model=_make(category, config),
    )


@register_model
def GPT2_large(args):
    category = "AutoModelForCausalLM"
    config = AutoConfig.from_pretrained("gpt2-large")
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=1024,
        model=_make(category, config),
    )


@register_model
def T5(args):
    category = "AutoModelForSeq2SeqLM"
    config = AutoConfig.from_pretrained("t5-small")
    return NS(
        category=category,
        config=config,
        train_length=1024,
        eval_length=2048,
        model=_make(category, config),
    )


@register_model
def T5_base(args):
    category = "AutoModelForSeq2SeqLM"
    config = AutoConfig.from_pretrained("t5-base")
    return NS(
        category=category,
        config=config,
        train_length=1024,
        eval_length=2048,
        model=_make(category, config),
    )


@register_model
def T5_large(args):
    category = "AutoModelForSeq2SeqLM"
    config = AutoConfig.from_pretrained("t5-large")
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=512,
        model=_make(category, config),
    )


@register_model
def Bart(args):
    category = "AutoModelForSeq2SeqLM"
    config = AutoConfig.from_pretrained("facebook/bart-base")
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=512,
        model=_make(category, config),
    )


@register_model
def Reformer(args):
    category = "AutoModelForMaskedLM"
    config = ReformerConfig()
    if not config.num_buckets:
        config.num_buckets = 128
    return NS(
        category=category,
        config=config,
        train_length=4096,
        eval_length=4096,
        model=_make(category, config),
    )


@register_model
def BigBird(args):
    category = "AutoModelForMaskedLM"
    config = BigBirdConfig(attention_type="block_sparse")
    return NS(
        category=category,
        config=config,
        train_length=1024,
        eval_length=4096,
        model=_make(category, config),
    )


@register_model
def Albert(args):
    category = "AutoModelForMaskedLM"
    config = AutoConfig.from_pretrained("albert-base-v2")
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=512,
        model=_make(category, config),
    )


@register_model
def DistilBert(args):
    category = "AutoModelForMaskedLM"
    config = AutoConfig.from_pretrained("distilbert-base-uncased")
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=512,
        model=_make(category, config),
    )


@register_model
def Longformer(args):
    category = "AutoModelForMaskedLM"
    config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
    return NS(
        category=category,
        config=config,
        train_length=1024,
        eval_length=4096,
        model=_make(category, config),
    )


@register_model
def Bert(args):
    category = "AutoModelForMaskedLM"
    config = BertConfig()
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=512,
        model=_make(category, config),
    )


@register_model
def Bert_large(args):
    category = "AutoModelForMaskedLM"
    config = BertConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16)
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=512,
        model=_make(category, config),
    )


@register_model
def Whisper(args):
    category = "AutoModelForAudioClassification"
    config = AutoConfig.from_pretrained("openai/whisper-tiny")
    return NS(
        category=category,
        config=config,
        train_length=config.max_target_positions,
        eval_length=config.max_target_positions,
        sampling_rate=16000,
        extractor_class=transformers.WhisperFeatureExtractor,
        model=_make(category, config),
    )


def dataset_ade20k(args, transform):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from collections import defaultdict
    import torch

    dataset = load_dataset("helenlu/ade20k", trust_remote_code=True)["train"]
    scenes = defaultdict(int)

    def collate_function(data):
        # {'image': <PIL.JpegImagePlugin.>, 'conditioning_image': <PIL.Pn>, 'text': 'bathroom'}
        images = []
        conditioning_images = []
        texts = []
        labels = []

        def get_label(txt):
            # Note: this is wrong for a real usecase because the label would change
            # depending on the shuffling
            nonlocal scenes
        
            label = scenes.get(txt)
            if label is None:
                label = len(scenes)
                scenes[text] = label
            return label

        for items in data:
            image = items["image"]
            conditioning_image = items["conditioning_image"]
            text = items["text"]
            label = get_label(text)

            texts.append(text)
            labels.append(label)
            images.append(transform(image, return_tensors="pt")["pixel_values"])
            conditioning_images.append(transform(conditioning_image, return_tensors="pt")["pixel_values"])

        return {
            "pixel_values": torch.cat(images),
            "conditioning_images": torch.cat(conditioning_images),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=collate_function
    )

    for i in loader:
        assert i['pixel_values'].shape == (args.batch_size, 3, 224, 224)
        print(i['pixel_values'].shape)
        print(i['conditioning_images'].shape)
        print(i['labels'].shape)
        break

    return loader


@register_model
def dinov2_large(args):
    category = "AutoModel"
    config = AutoConfig.from_pretrained("facebook/dinov2-large")


    def criterion(model_output, dataloader_input):
        mask = dataloader_input["conditioning_images"]
        print(model_output)
        return 0

    processor = transformers.AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=1024,
        model=_make(category, config),
        transform=processor,
        dataloader = dataset_ade20k(args, processor),
        model_inputs=lambda x: {"pixel_values": x["pixel_values"]},
        criterion=criterion
    )







@register_model
def dinov20_giant(args):
    category = "AutoModel"
    config = transformers.Dinov2Config("facebook/dinov2-giant")
    processor = transformers.AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=1024,
        model=_make(category, config),
        transform=processor,
        dataloader=dataset_ade20k(args, processor),
        model_inputs=lambda x: {"pixel_values": x["pixel_values"]}
    )
