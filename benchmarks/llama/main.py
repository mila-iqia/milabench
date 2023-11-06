
import json
import os
import argparse
import time
import sys
import multiprocessing

import torch

from voir.smuggle import SmuggleWriter
from voir.instruments.gpu import get_gpu_info

root = os.path.dirname(__file__)


def available_models():
    models = dict()

    for size in ("7b", "13b", "70b"):
        models[f'llama2-{size}'] = {
            "name": f"meta-llama/Llama-2-{size}-chat-hf",
            "config": f"llama2_{size}_chat_hf.config"
        }
        
    return models


def _worker(state, queue, func, delay):
    import time

    while state['running']:
        queue.put(func())
        time.sleep(delay)
        
class Monitor:
    def __init__(self, delay, func):
        self.manager = multiprocessing.Manager()
        self.state = self.manager.dict()
        self.state['running'] = True
        self.results = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=_worker, 
            args=(self.state, self.results, func, delay),
        )
        
    def start(self):
        self.process.start()
        
    def stop(self):
        self.state['running'] = False
        self.process.join()


def setupvoir():
    # wtf this do
    data_file = SmuggleWriter(sys.stdout)
    # data_file = sys.stdout
    
    def log(data):
        if data_file is not None:
            data["t"] = time.time()
            print(json.dumps(data), file=data_file)
            
            while not monitor.results.empty():
                print(json.dumps(monitor.results.get()), file=data_file)
        
    def monitor_fn():
        data = {
            gpu["device"]: {
                "memory": [
                    gpu["memory"]["used"], 
                    gpu["memory"]["total"],
                ],
                "load": gpu["utilization"]["compute"],
                "temperature": gpu["temperature"],
                "power": gpu["power"]
            }
            for gpu in get_gpu_info()["gpus"].values()
        }
        return {"task": "main", "gpudata": data, "t": time.time()}
        
    monitor = Monitor(0.5, monitor_fn)
    monitor.start()
    return log, monitor


class WrappedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.count = 0

    def __call__(self, *args, **kwargs):
        input_ids = self.tokenizer(*args, **kwargs)
        self.count = len(input_ids)
        return input_ids

    def __getattr__(self, attr):
        if hasattr(self.tokenizer, attr):
            method = getattr(self.tokenizer, attr)
            return method
        else:
            raise AttributeError(f"'{type(self.tokenizer).__name__}' object has no attribute '{attr}'")


def huggingface_main(args, model, config):
    # Huggingface imported AFTER setup
    import transformers
    from transformers import LlamaForCausalLM, LlamaTokenizerFast
    from transformers.models.llama.configuration_llama import LlamaConfig

    from datasets import load_dataset
    
    # Dataset here
    dataset = load_dataset(
        "wikitext", 
        "wikitext-103-v1"
    )
    
    # LLAMA tokenizer official tokenizer is hidden behind a login
    tokenizer = WrappedTokenizer(
        LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    )

    # Prepare is done
    if args.prepare:
        return 0
    
    # We do not download LLAMA because it takes too long
    # we just instantiate an untrained one
    model = LlamaForCausalLM(LlamaConfig.from_dict(config))
        
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        tokenizer=tokenizer,
    )
    
    in_token_count = 0
    out_token_count = 0
    
    start = time.time()
    
    log, monitor = setupvoir()

    for entry in dataset["train"]:
        text = entry["text"].strip()
        
        # Titles
        if text == "" or text.startswith(" = "):
            continue
        
        
        sequences = pipeline(
            text,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=400,
        )
    
        for seq in sequences:
            out_token_count += len(seq["generated_text"])

        in_token_count += tokenizer.count
        total = out_token_count + in_token_count
        
        elapsed = time.time() - start
        print(total / elapsed)
        
        if total > 100:
            out_token_count = 0
            in_token_count = 0
            start = time.time()
            
            if log is not None:
                log({
                    "task": "train",
                    "rate": total / elapsed,
                    "units": "Tok/s"
                })
            

    monitor.stop()

def main():
    models = available_models()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama2-7b", choices=models.keys())
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cache", required=True, type=str)
    
    #
    args = parser.parse_args()
    os.environ["XDG_CACHE_HOME"] = str(args.cache)
    
    settings = models[args.model]
    model, config = settings["name"], settings["config"]
    
    with open(os.path.join(root, 'config', config), 'r') as file:
        config = json.load(file)

    return huggingface_main(args, model, config)



if __name__ == "__main__":
    main()
