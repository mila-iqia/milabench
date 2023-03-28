from types import SimpleNamespace as NS

import transformers
from transformers import AutoConfig, BertConfig, BigBirdConfig, ReformerConfig

models = {}


def register_model(fn):
    models[fn.__name__] = fn
    return fn


@register_model
def Opt350m():
    config = AutoConfig.from_pretrained("facebook/opt-350m")
    return NS(
        config=config,
        train_length=config.max_position_embeddings,
        eval_length=config.max_position_embeddings,
        model=transformers.AutoModelForCausalLM.from_config(config),
    )


@register_model
def GPT2():
    config = AutoConfig.from_pretrained("gpt2")
    return NS(
        config=config,
        train_length=512,
        eval_length=1024,
        model=transformers.AutoModelForCausalLM.from_config(config),
    )


@register_model
def GPT2_large():
    config = AutoConfig.from_pretrained("gpt2-large")
    return NS(
        config=config,
        train_length=512,
        eval_length=1024,
        model=transformers.AutoModelForCausalLM.from_config(config),
    )


@register_model
def T5():
    config = AutoConfig.from_pretrained("t5-small")
    return NS(
        config=config,
        train_length=1024,
        eval_length=2048,
        model=transformers.AutoModelForSeq2SeqLM.from_config(config),
    )


@register_model
def T5_base():
    config = AutoConfig.from_pretrained("t5-base")
    return NS(
        config=config,
        train_length=1024,
        eval_length=2048,
        model=transformers.AutoModelForSeq2SeqLM.from_config(config),
    )


@register_model
def T5_large():
    config = AutoConfig.from_pretrained("t5-large")
    return NS(
        config=config,
        train_length=512,
        eval_length=512,
        model=transformers.AutoModelForSeq2SeqLM.from_config(config),
    )


@register_model
def Bart():
    config = AutoConfig.from_pretrained("facebook/bart-base")
    return NS(
        config=config,
        train_length=512,
        eval_length=512,
        model=transformers.AutoModelForSeq2SeqLM.from_config(config),
    )


@register_model
def Reformer():
    config = ReformerConfig()
    if not config.num_buckets:
        config.num_buckets = 128
    return NS(
        config=config,
        train_length=4096,
        eval_length=4096,
        model=transformers.AutoModelForMaskedLM.from_config(config),
    )


@register_model
def BigBird():
    config = BigBirdConfig(attention_type="block_sparse")
    return NS(
        config=config,
        train_length=1024,
        eval_length=4096,
        model=transformers.AutoModelForMaskedLM.from_config(config),
    )


@register_model
def Albert():
    config = AutoConfig.from_pretrained("albert-base-v2")
    return NS(
        config=config,
        train_length=512,
        eval_length=512,
        model=transformers.AutoModelForMaskedLM.from_config(config),
    )


@register_model
def DistilBert():
    config = AutoConfig.from_pretrained("distilbert-base-uncased")
    return NS(
        config=config,
        train_length=512,
        eval_length=512,
        model=transformers.AutoModelForMaskedLM.from_config(config),
    )


@register_model
def Longformer():
    config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
    return NS(
        config=config,
        train_length=1024,
        eval_length=4096,
        model=transformers.AutoModelForMaskedLM.from_config(config),
    )


@register_model
def Bert():
    config = BertConfig()
    return NS(
        config=config,
        train_length=512,
        eval_length=512,
        model=transformers.AutoModelForMaskedLM.from_config(config),
    )


@register_model
def Bert_large():
    config = BertConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16)
    return NS(
        config=config,
        train_length=512,
        eval_length=512,
        model=transformers.AutoModelForMaskedLM.from_config(config),
    )
