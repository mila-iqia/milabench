from dataclasses import dataclass

from datasets import load_dataset, Audio
from argklass import ArgumentParser
from argklass.arguments import argument
import torch
import torchcompat.core as accelerator
from torch.utils.data import DataLoader


whisper_defaults_generation_args = {
    # "max_new_tokens": 448,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    # zlib compression ratio threshold (in token space)
    "compression_ratio_threshold": 1.35,  
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}

flux_default_generation_args = {
    "height": 256,
    "width": 256,
    "guidance_scale": 3.5,
    "num_inference_steps": 50,
    "max_sequence_length": 512,
    "generator": torch.Generator(accelerator.device_type).manual_seed(0)
}

chat_default_generation_args = {

}


class InferenceBenchmark:
    def __init__(self):
        self.raise_stop = False
        self.custom_step = False

    def get_batch_size(self, item):
        return len(item)

    def prepare_voir(self, args):
        from benchmate.observer import BenchObserver
        from benchmate.monitor import bench_monitor
        
        observer = BenchObserver(
            accelerator.Event, 
            earlystop=65,
            batch_size_fn=self.get_batch_size,
            raise_stop_program=self.raise_stop,
            stdout=True,
        )

        return observer, bench_monitor

    def load_model(self, args, device):
        pass

    def transform(self, item):
        return item

    def collate(self, group):
        batch = []
        for item in group:
            batch.append(self.transform(item))
        return batch

    def load_dataset(self, observer, args):
        # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        dataset = load_dataset(
            args.dataset,
            name=args.subset,  # Subset
            split=args.split,  # Split
        )

        return observer.loader(self.dataloader(dataset, args), custom_step=self.custom_step)

    def dataloader(self, dataset, args):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=self.collate
        )

    def run(self, pipe, batch, kwargs):
        return pipe(batch, **kwargs, batch_size=len(batch))


class WhisperBenchmark(InferenceBenchmark):
    def load_model(self, args, device):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model, 
            dtype=args.dtype, 
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        model.to(device)

        processor = AutoProcessor.from_pretrained(args.model)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=args.dtype,
            device=device,
        )

        kwargs = dict(args.kwargs) or whisper_defaults_generation_args
        return pipe, kwargs

    def get_batch_size(self, x):
        # Audio is Samples/sec
        # Image is Img/Sec
        # Chat is Token/Sec
        samples = 0
        for item in x:
            # 16000 is usually the sampling rate
            # This is used to reduce the score so it is more readable
            # also this makes the score be the number of seconds
            # processed in one seconds
            #
            samples += item["array"].shape[0] / 16000
        
        return samples

    def transform(self, item):
        audio = item["audio"]
        data = audio.get_all_samples()
        array = data.data.mean(dim=0)
        return {
            "array": array,
            "sampling_rate": data.sample_rate
        }


class FluxBenchmark(InferenceBenchmark):
    def __init__(self):
        super().__init__()
    
        self.i = 0
        self.dataset = None
        self.bs = 0
        self.raise_stop = False
        self.custom_step = True

    def get_batch_size(self, item):
        self.bs = len(item)
        return self.bs

    def load_model(self, args, device):
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            args.model, 
            torch_dtype=torch.bfloat16,
            device_map="cuda"
            # Unexpected by FLux
            # dtype=args.dtype,
            # device=device,
        ) # .to("cuda")

        if False:
            models = {
                'transformer': pipeline.transformer,
                'scheduler': pipeline.scheduler,
                'vae': pipeline.vae,
                'text_encoder': pipeline.text_encoder,
                'text_encoder_2': pipeline.text_encoder_2,
                'tokenizer': pipeline.tokenizer,
                'tokenizer_2': pipeline.tokenizer_2,
            }

            for model in models:
                pass

        # pipeline.transformer.to(memory_format=torch.channels_last)
        # pipeline.vae.to(memory_format=torch.channels_last)
        # pipeline.transformer.enable_forward_chunking(chunk_size=1, dim=1)
        # torch.compile(pipeline.)

        # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.enable_model_cpu_offload()

        kwargs = dict(args.kwargs) or flux_default_generation_args
        return pipe, kwargs

    def load_dataset(self, observer, args):
        base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
        num_shards = 10  # Number of webdataset tar files
        urls = [base_url.format(i=i) for i in range(num_shards)]
        dataset = load_dataset(
            "webdataset", 
            data_files={"train": urls}, 
            split="train", 
            streaming=False
        )

        self.dataset = observer.loader(self.dataloader(dataset, args), custom_step=self.custom_step)
        return self.dataset

    def transform(self, item):
        p = item["json"]["prompt"]
        return p[:min(len(p), 70)]

    def collate(self, group):
        batch = []
        for item in group:
            batch.append(self.transform(item))
        
        self.bs = len(batch)
        return batch

    def on_step(self, pipe, step: int, timestep: int, kwargs):
        self.dataset.acc_batch_size = self.bs

        should_stop = self.dataset.step()
        # Here we measure the time it takes to do a denoising step
        # not to generate a whole image because it takes too long
        self.dataset.acc_batch_size = self.bs
        return {}

    def run(self, pipe, batch, kwargs):
        output = pipe(batch, 
            callback_on_step_end_tensor_inputs=[],
            callback_on_step_end=self.on_step, 
            **kwargs
        )
        
        # for img in output.images:
        #     img.save(f"/tmp/data/flux_{self.i}.png")
        #     self.i += 1
        return output




class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tok_in = 0

    def __call__(self, *args, **kwargs):
        tensor = self.tokenizer(*args, **kwargs)
        shape = tensor["input_ids"].shape
        self.tok_in += shape[0] * shape[1]
        return tensor


class ChatBenchmark(InferenceBenchmark):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.tok_per_sec = True

    def transform(self, item):
        return item["problem"]

    def get_batch_size(self, item):
        return len(item) * 100

    def load_dataset(self, observer, args):
        dataset = load_dataset(
            args.dataset,
            name=args.subset,  # Subset
            split=args.split,  # Split
        )

        self.dataset = observer.loader(self.dataloader(dataset, args), custom_step=self.tok_per_sec)
        return self.dataset

    def load_model(self, args, device):
        import transformers
 

        # TextGenerationPipeline
        pipe = transformers.pipeline(
            "text-generation",
            model=args.model,
            model_kwargs={"dtype": torch.bfloat16},
            device_map="auto",
        )

        pipe.tokenizer = TokenizerWrapper(pipe.tokenizer)

        def async_generation(inputs, **kwargs):
            from transformers.generation.streamers import TextIteratorStreamer
            from threading import Thread
            import time

            streamer = TextIteratorStreamer(pipe.tokenizer)
            model = pipe.model

            generation_kwargs = {
                **inputs, 
                "streamer": streamer, 
                **kwargs,
                # max_new_tokens=20
            }

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            start = time.time()
            for txt in streamer:
                yield start, time.time(), txt

        kwargs = dict(args.kwargs) or chat_default_generation_args
        return pipe, kwargs

    def run(self, pipe, batch, kwargs):
        outputs = pipe(batch, return_tensors=True, **kwargs)

        if self.tok_per_sec:
            tok_out = 0
            for out in outputs:
                for o in out:
                    tok_out += len(o['generated_token_ids'])
            
            tok_tot = pipe.tokenizer.tok_in + tok_out
            self.dataset.step(tok_tot)
            pipe.tokenizer.tok_in = 0
        
        return outputs


def load_benchmark(argv):
    match argv.mode:
        case "whisper":
            return WhisperBenchmark()

        case "flux":
            return FluxBenchmark()
    
        case "chat":
            return ChatBenchmark()
    
    raise RuntimeError(f"Benchmark {argv.mode} does not exist")


def parse_kv(s):
    key, value = s.split("=", 1)
    return key, value


@dataclass
class Arguments:
    mode: str = None
    dataset: str = None
    split: str = None
    subset: str = None
    model: str = None
    batch_size: int = 16
    kwargs: list = argument(default_factory=list, nargs="+", default=[], type=parse_kv)
    dtype: str = "bfloat16"
    multi_gpu: bool = False
    prepare: bool = False


def main(argv=None):
    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args(argv)

    bench = load_benchmark(args) 
    observer, monitor = bench.prepare_voir(args)
    device = accelerator.fetch_device(0)
    
    with monitor():
        with torch.no_grad():
            dataset = bench.load_dataset(observer, args)

            pipe, kwargs = bench.load_model(args, device)

            if args.prepare:
                return 0

            # We cannot wrap the dataset with our timed loader anymore
            # dataset = setup_dataset(args)
            # output = pipe(dataset, **kwargs, batch_size=args.batch_size)

            # Here it still works
            for batch in dataset:
                output = bench.run(pipe, batch, kwargs)


# MultiGPU Setup (?)
#   Not worth doing that might as well just launch N same bench
#   Split the batch across GPUs
#       result = pipe(texts[accelerator.process_index::accelerator.num_processes])
#   

if __name__ == "__main__":
    # milabench run --config /home/mila/d/delaunap/scratch/milabench/benchmarks/inference/dev.yaml --base /tmp/data/ --use-current-env --select whisper-transcribe-single
    # milabench run --config /home/mila/d/delaunap/scratch/milabench/benchmarks/inference/dev.yaml --base /tmp/data/ --use-current-env --select txt-to-image-gpus
    # milabench run --config dev.yaml --base /tmp/data/ --use-current-env --select llm-chat-completion


    main()
