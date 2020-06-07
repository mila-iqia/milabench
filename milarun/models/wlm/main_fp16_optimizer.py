import os
import argparse
import time
import math
import torch
import torch.nn as nn

from . import model as model_module

from coleo import Argument, default
from milarun.lib import coleo_main, init_torch, iteration_wrapper, dataloop


try:
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


@coleo_main
def main(exp):

    # dataset to use
    dataset: Argument & str

    # type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
    model_name: Argument & str = default('LSTM')

    # size of word embeddings
    emsize: Argument & int = default(200)

    # number of hidden units per layer
    nhid: Argument & int = default(200)

    # number of layers
    nlayers: Argument & int = default(2)

    # initial learning rate
    lr: Argument & float = default(20)

    # gradient clipping
    clip: Argument & float = default(0.25)

    # upper epoch limit
    epochs: Argument & int = default(40)

    # sequence length
    bptt: Argument & int = default(35)

    # dropout applied to layers (0 = no dropout)
    dropout: Argument & float = default(0.2)

    # tie the word embedding and softmax weights
    tied: Argument & bool = default(False)

    # report interval
    log_interval: Argument & int = default(200)

    # Run model in pseudo-fp16 mode (fp16 storage fp32 math).
    fp16: Argument & bool = default(True)

    # Static loss scale, positive power of 2 values can improve fp16 convergence.
    static_loss_scale: Argument & float = default(128.0)

    # Use dynamic loss scaling.
    # If supplied, this argument supersedes --static-loss-scale.
    dynamic_loss_scale: Argument & bool = default(False)

    # path to save the final model
    save: Argument & str = default(None)

    # path to export the final model in onnx format
    batch_size: Argument & int = default(64)

    # Maximum count before stopping
    max_count: Argument & int = default(1000)

    # Number of seconds for sampling items/second
    sample_duration: Argument & float = default(0.5)

    torch_settings = init_torch()
    device = torch_settings.device


    ###############################################################################
    # Load data
    ###############################################################################

    # Ensure that the dictionary length is a multiple of 8,
    # so that the decoder's GEMMs will use Tensor Cores.
    corpus = exp.get_dataset(dataset, pad_to_multiple_of=8).corpus

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        if torch_settings.cuda:
            data = data.cuda()
        return data

    eval_batch_size = 10
    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)

    if fp16 and torch_settings.cuda:
        if ntokens%8 != 0:
            print("Warning: the dictionary size (ntokens = {}) should be a multiple of 8 to ensure "
                "Tensor Core use for the decoder's GEMMs.".format(ntokens))
        if emsize%8 != 0 or nhid%8 != 0 or batch_size%8 != 0:
            print("Warning: emsize = {}, nhid = {}, batch_size = {} should all be multiples of 8 "
                "to ensure Tensor Core use for the RNN's GEMMs.".format(
                emsize, nhid, batch_size))

    model = model_module.RNNModel(model_name, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)

    if torch_settings.cuda and fp16:
        model.type(torch.cuda.HalfTensor)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    ###############################################################################
    # Create the FP16_Optimizer instance
    ###############################################################################

    if fp16 and torch_settings.cuda:
        # If dynamic_loss_scale is False, static_loss_scale will be used.
        # If dynamic_loss_scale is True, it will take precedence over static_loss_scale.
        optimizer = FP16_Optimizer(optimizer,
                                static_loss_scale = static_loss_scale,
                                dynamic_loss_scale = dynamic_loss_scale)

    ###############################################################################
    # Training code
    ###############################################################################


    def repackage_hidden(h):
        """Detaches hidden states from their history."""
        if torch.is_tensor(h):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


    # get_batch subdivides the source data into chunks of length bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target


    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, ntokens)
                #total loss can overflow if accumulated in fp16.
                total_loss += len(data) * criterion(output_flat, targets).data.float()
                hidden = repackage_hidden(hidden)
        return to_python_float(total_loss) / len(data_source)


    def train(chrono):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        for batch, i in enumerate(range(0, len(train_data), bptt)):
            if chrono.done():
                break
            with chrono(count=batch_size) as it:
                data, targets = get_batch(train_data, i)
                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden = repackage_hidden(hidden)
                model.zero_grad()
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, ntokens), targets)

                # Clipping gradients helps prevent the exploding gradient problem in RNNs / LSTMs.
                if fp16 and torch_settings.cuda:
                    optimizer.backward(loss)
                    optimizer.clip_master_grads(clip)
                else:
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    # apex.fp16_utils.clip_grad_norm selects between "torch.nn.utils.clip_grad_norm" 
                    # and "torch.nn.utils.clip_grad_norm_" based on Pytorch version.  
                    # It's not FP16-specific, just a small fix to avoid deprecation warnings.
                    clip_grad_norm(model.parameters(), clip)

                optimizer.step()

                it.log(loss=loss.item())
                total_loss += loss.data

                # if batch % args.log_interval == 0 and batch > 0:
                #     cur_loss = to_python_float(total_loss) / args.log_interval
                #     elapsed = time.time() - start_time
                #     print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                #         'loss {:5.2f} | ppl {:8.2f}'.format(
                #             epoch, batch, len(train_data) // args.bptt, lr,
                #             elapsed * 1000 / args.log_interval, cur_loss, math.exp(min(cur_loss, 20))))
                #     total_loss = 0
                #     start_time = time.time()

    # Loop over epochs.
    best_val_loss = None

    chrono = exp.chronos.create(
        "train",
        type="rate",
        sync=torch_settings.sync,
        sample_duration=sample_duration,
        max_count=max_count,
    )

    while not chrono.done():
        train(chrono)
        val_loss = evaluate(val_data)

        exp.metrics["val_loss"] = val_loss

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
