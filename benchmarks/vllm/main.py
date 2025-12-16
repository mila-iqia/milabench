from argparse import ArgumentParser
import subprocess
import warnings
import threading
import time
import signal
import os

import numpy as np
import torchcompat.core as accelerator
from vllm.benchmarks.serve import SampleRequest, RequestFuncOutput, PreTrainedTokenizerBase, BenchmarkMetrics, MILLISECONDS_TO_SECONDS_CONVERSION
import vllm.benchmarks.datasets as datasets

push_metric = None


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    """Calculate the metrics for the benchmark.

    Args:
        input_requests: The input requests.
        outputs: The outputs of the requests.
        dur_s: The duration of the benchmark.
        tokenizer: The tokenizer to use.
        selected_percentiles: The percentiles to select.
        goodput_config_dict: The goodput configuration.

    Returns:
        A tuple of the benchmark metrics and the actual output lengths.
    """
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)

            push_metric(ttfts=outputs[i].ttft, unit="ms")
            push_metric(e2els=outputs[i].latency, unit="ms")
            
            if len(outputs[i].itl) > 0:
                push_metric(itl=sum(outputs[i].itl)/len(outputs[i].itl), unit="ms")

            push_metric(tpot=outputs[i].tpot, unit="ms")
            push_metric(input_tok=input_requests[i].prompt_len, unit="count")
            push_metric(output_tok=output_len, unit="count")

            tok_s = (input_requests[i].prompt_len + output_len) / outputs[i].latency
            push_metric(rate=tok_s, unit="tok/s")
            
            completed += 1
        else:
            actual_output_lens.append(0)


class GPQADiamond(datasets.HuggingFaceDataset):
    IS_MULTIMODAL = False
    SUPPORTED_DATASET_PATHS = {'hendrydong/gpqa_diamond'}

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list:
        sampled_requests = []
        ind = 0
        dynamic_output = output_len is None

        for item in self.data["test"]:
            if len(sampled_requests) >= num_requests:
                break

            prompt, completion = item["problem"], item["solution"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids

            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len

            assert isinstance(output_len, int) and output_len > 0

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=request_id_prefix + str(ind),
                )
            )
            ind += 1
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


def benchmark(argv):
    # vllm bench serve --model meta-llama/Meta-Llama-3-8B-Instruct --request-rate inf --dataset-name random --label milabench --backend openai --num-prompts 1000

    # vllm bench serve                                      \
    #     --backend openai                                  \
    #     --label milabench                                 \
    #     --model <your_model>                              \
    #     --dataset-name <dataset_name. Default 'random'>   \
    #     --request-rate inf                                \
    #     --num-prompts 1000
    import vllm.benchmarks.serve as bench
    import vllm.benchmarks.datasets as datasets

    
    # datasets.InstructCoderDataset
    # datasets.BlazeditDataset
    #       Coding Task
    # https://huggingface.co/datasets/likaixin/InstructCoder

    # datasets.MTBenchDataset
    #       Open ended writing
    #  https://huggingface.co/datasets/philschmid/mt-bench

    # datasets.AIMODataset
    #       reasoning questions


    def open_dataset(dataset_cls, args, tokenizer):
        hf_kwargs = {}
        return dataset_cls(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
            no_stream=args.no_stream,
            hf_name=args.hf_name,
            # missing arg ?
            # disable_shuffle=args.disable_shuffle,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
            # missing arg ?
            # skip_chat_template=args.skip_chat_template,
            **hf_kwargs,
        )
           

    original_get_samples = bench.get_samples
    def get_samples(args, tokenizer):
        match args.hf_name:
            case "openslr/librispeech_asr":
                return open_dataset(datasets.ASRDataset, args, tokenizer)
            
            case "hendrydong/gpqa_diamond":
                return open_dataset(GPQADiamond, args, tokenizer)

            case _:
                return original_get_samples(args, tokenizer)

    bench.get_samples = get_samples

    original = bench.calculate_metrics

    def new_calculate_metrics(*args, **kwargs):
        calculate_metrics(*args, **kwargs)
        return original(*args, **kwargs)

    bench.calculate_metrics = new_calculate_metrics

    parser = ArgumentParser()
    bench.add_cli_args(parser)

    print("BENCH:", " ".join(['vllm', 'bench', 'serve'] + argv))
    args = parser.parse_args(argv)

    bench.main(args)
    print("FINISHED")


def prepare_voir():
    global push_metric 

    from benchmate.observer import BenchObserver
    from benchmate.monitor import bench_monitor

    observer = BenchObserver(
        accelerator.Event, 
        earlystop=65,
        batch_size_fn=lambda x: len(x[0]),
        raise_stop_program=False,
        stdout=True,
    )

    push_metric = observer.record_metric
    return observer, bench_monitor



class InferenceServerError(BaseException):
    pass


def monitor_process(proc):
    """Wait for subprocess to fail, then terminate the main process."""
    while True:
        ret = proc.poll()
        if ret is not None:
            if ret != 0:
                print(f"\n[ERROR] vLLM server exited with code {ret}", file=sys.stderr)
                os._exit(2)
            break
        time.sleep(0.5)


def inference_server(argv):
    # vllm serve meta-llama/Meta-Llama-3-8B-Instruct --dtype bfloat16 

    server_args = ["vllm", "serve"] + argv

    print("SERVER:", " ".join(server_args))

    proc = subprocess.Popen(server_args)

    monitor = threading.Thread(target=monitor_process, args=(proc,), daemon=True)
    monitor.start()

    return proc


def split_args(argv):
    for i, arg in enumerate(argv):
        if arg == "--":
            break
    
    server_argv = argv[1:i]
    bench_argv = argv[(i + 1):]

    args = []
    for arg in server_argv:
        # Voir already has a --config argument so we have to rename it
        if arg == "--wth-config":
            args.append("--config")
        else:
            args.append(arg)
    server_argv = args

    return server_argv, bench_argv


def main(argv):
    global push_metric 

    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # parser.add_argument("--prepare", action="store_true", default=True)
    # args, argv = parser.parse_known_args(argv)

    server_argv, bench_argv = split_args(argv)

    observer, bench_monitor= prepare_voir()

    with bench_monitor() as log:
        try:
            with inference_server(server_argv) as proc:
                try:
                    benchmark(bench_argv)
                finally:
                    proc.kill()
        except Exception:
            raise


if __name__ == "__main__":
    import sys

    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))

    # python -c "import debugpy; debugpy.connect(('localhost', 5678')); debugpy.breakpoint()"

    main(sys.argv)
