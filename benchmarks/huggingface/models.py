from types import SimpleNamespace as NS

import transformers
from transformers import AutoConfig, BertConfig, BigBirdConfig, ReformerConfig

models = {}


def register_model(fn):
    models[fn.__name__] = fn
    return fn


def _make(category, config):
    return getattr(transformers, category).from_config(config)


@register_model
def Opt350m():
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
def GPT2():
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
def GPT2_large():
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
def T5():
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
def T5_base():
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
def T5_large():
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
def Bart():
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
def Reformer():
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
def BigBird():
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
def Albert():
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
def DistilBert():
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
def Longformer():
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
def Bert():
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
def Bert_large():
    category = "AutoModelForMaskedLM"
    config = BertConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16)
    return NS(
        category=category,
        config=config,
        train_length=512,
        eval_length=512,
        model=_make(category, config),
    )
