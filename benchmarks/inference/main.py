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
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 50,
    "max_sequence_length": 512,
    "generator": torch.Generator("cpu").manual_seed(0)
}

chat_default_generation_args = {

}


class InferenceBenchmark:
    def get_batch_size(self, item):
        return len(item)

    def prepare_voir(self, args):
        from benchmate.observer import BenchObserver
        from benchmate.monitor import bench_monitor
        
        observer = BenchObserver(
            accelerator.Event, 
            earlystop=65,
            batch_size_fn=self.get_batch_size,
            raise_stop_program=False,
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

    def load_dataset(self, args):
        # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        dataset = load_dataset(
            args.dataset,
            name=args.subset,  # Subset
            split=args.split,  # Split
        )

        return self.dataloader(dataset, args)

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
    def load_model(self, args, device):
        from diffusers import FluxPipeline

        # FluxFillPipeline
        # FluxControlPipeline
        

        pipe = FluxPipeline.from_pretrained(
            args.model, 
            torch_dtype=torch.bfloat16,
            # Unexpected by FLux
            # dtype=args.dtype,
            # device=device,
        ).to("cuda")

        # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.enable_model_cpu_offload()

        kwargs = dict(args.kwargs) or flux_default_generation_args
        return pipe, kwargs

    def load_dataset(self, args):
        base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
        num_shards = 10  # Number of webdataset tar files
        urls = [base_url.format(i=i) for i in range(num_shards)]
        dataset = load_dataset(
            "webdataset", 
            data_files={"train": urls}, 
            split="train", 
            streaming=False
        )

        return self.dataloader(dataset, args)

    def transform(self, item):
        print(item)
        return item["json"]["prompt"]

    def run(self, pipe, batch, kwargs):
        return pipe(batch, **kwargs)


class ChatBenchmark(InferenceBenchmark):
    def load_model(self, args, device):
        pipeline = transformers.pipeline(
            "text-generation",
            model=args.model,
            model_kwargs={"dtype": torch.bfloat16},
            device_map="auto",
        )

        kwargs = dict(args.kwargs) or chat_default_generation_args
        return pipe, kwargs


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


def main(argv=None):
    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args(argv)

    bench = load_benchmark(args) 
    observer, monitor = bench.prepare_voir(args)
    device = accelerator.fetch_device(0)
    
    with monitor():
        with torch.no_grad():
            dataset = observer.loader(bench.load_dataset(args))

            pipe, kwargs = bench.load_model(args, device)

            # We cannot wrap the dataset with out timed loader anymore
            # dataset = setup_dataset(args)
            # output = pipe(dataset, **kwargs, batch_size=args.batch_size)
            for batch in dataset:
                output = bench.run(pipe, batch, kwargs)


if __name__ == "__main__":
    # milabench run --config /home/mila/d/delaunap/scratch/milabench/benchmarks/inference/dev.yaml --base /tmp/data/ --use-current-env --select whisper-transcribe-single
    # milabench run --config /home/mila/d/delaunap/scratch/milabench/benchmarks/inference/dev.yaml --base /tmp/data/ --use-current-env --select txt-to-image-gpus

    main()
