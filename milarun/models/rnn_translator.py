import argparse
import os
import logging
from ast import literal_eval
import subprocess

import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
import torch.optim

from mlperf_compliance import mlperf_log

from .seq2seq import models
from .seq2seq.train.smoothing import LabelSmoothing
from .seq2seq.data.dataset import ParallelDataset
from .seq2seq.data.tokenizer import Tokenizer
from .seq2seq.utils import setup_logging
from .seq2seq.data import config
from .seq2seq.train import trainer as trainers
from .seq2seq.inference.inference import Translator

from coleo import Argument, default
from milarun.lib import coleo_main, init_torch, iteration_wrapper, dataloop


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=100, type=int, help="seed to use")
    parser.add_argument("--cuda", action="store_true", help="whether to use CUDA")
    parser.add_argument("--workers", default=8, type=int, help="number of workers")
    parser.add_argument("--max-count", default=1000, type=int, help="Maximum count before stopping")
    parser.add_argument("--sample-duration", default=0.5, type=float, help="Number of seconds for sampling items/second")
    parser.add_argument("--batch-size", default=64, type=int, help="batch size")

    # dataset
    dataset = parser.add_argument_group('dataset setup')
    # dataset.add_argument('--dataset-dir', default=None, required=True,
    #                      help='path to directory with training/validation data')
    dataset.add_argument('--dataset', default=None, required=True,
                         help='dataset to use')
    dataset.add_argument('--max-size', default=None, type=int,
                         help='use at most MAX_SIZE elements from training \
                        dataset (useful for benchmarking), by default \
                        uses entire dataset')

    # results
    results = parser.add_argument_group('results setup')
    results.add_argument('--results-dir', default='../results',
                         help='path to directory with results, it it will be \
                        automatically created if does not exist')
    results.add_argument('--save', default=None,
                         help='defines subdirectory within RESULTS_DIR for \
                        results from this training run')
    results.add_argument('--print-freq', default=10, type=int,
                         help='print log every PRINT_FREQ batches')

    # model
    model = parser.add_argument_group('model setup')
    model.add_argument('--model-config',
                       default="{'hidden_size': 1024,'num_layers': 4, \
                        'dropout': 0.2, 'share_embedding': True}",
                       help='GNMT architecture configuration')
    model.add_argument('--smoothing', default=0.1, type=float,
                       help='label smoothing, if equal to zero model will use \
                        CrossEntropyLoss, if not zero model will be trained \
                        with label smoothing loss based on KLDivLoss')

    # setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp32', choices=['fp32', 'fp16'],
                         help='arithmetic type')
    general.add_argument('--disable-eval', action='store_true', default=False,
                         help='disables validation after every epoch')

    cudnn_parser = general.add_mutually_exclusive_group(required=False)
    cudnn_parser.add_argument('--cudnn', dest='cudnn', action='store_true',
                              help='enables cudnn (use \'--no-cudnn\' to disable)')
    cudnn_parser.add_argument('--no-cudnn', dest='cudnn', action='store_false',
                              help=argparse.SUPPRESS)
    cudnn_parser.set_defaults(cudnn=True)

    # training
    training = parser.add_argument_group('training setup')
    # training.add_argument('--epochs', default=8, type=int,
    #                       help='number of total epochs to run')
    training.add_argument('--optimization-config',
                          default="{'optimizer': 'Adam', 'lr': 5e-4}", type=str,
                          help='optimizer config')
    training.add_argument('--grad-clip', default=5.0, type=float,
                          help='enabled gradient clipping and sets maximum \
                        gradient norm value')
    training.add_argument('--max-length-train', default=50, type=int,
                          help='maximum sequence length for training')
    training.add_argument('--min-length-train', default=0, type=int,
                          help='minimum sequence length for training')
    training.add_argument('--target-bleu', default=None, type=float,
                          help='target accuracy')

    bucketing_parser = training.add_mutually_exclusive_group(required=False)
    bucketing_parser.add_argument('--bucketing', dest='bucketing', action='store_true',
                             help='enables bucketing (use \'--no-bucketing\' to disable)')
    bucketing_parser.add_argument('--no-bucketing', dest='bucketing', action='store_false',
                             help=argparse.SUPPRESS)
    bucketing_parser.set_defaults(bucketing=True)

    # validation
    validation = parser.add_argument_group('validation setup')
    validation.add_argument('--eval-batch-size', default=32, type=int,
                            help='batch size for validation')
    validation.add_argument('--max-length-val', default=150, type=int,
                            help='maximum sequence length for validation')
    validation.add_argument('--min-length-val', default=0, type=int,
                            help='minimum sequence length for validation')

    validation.add_argument('--beam-size', default=5, type=int,
                        help='beam size')
    validation.add_argument('--len-norm-factor', default=0.6, type=float,
                        help='length normalization factor')
    validation.add_argument('--cov-penalty-factor', default=0.1, type=float,
                        help='coverage penalty factor')
    validation.add_argument('--len-norm-const', default=5.0, type=float,
                        help='length normalization constant')


    # checkpointing
    checkpoint = parser.add_argument_group('checkpointing setup')
    checkpoint.add_argument('--start-epoch', default=0, type=int,
                            help='manually set initial epoch counter')
    checkpoint.add_argument('--resume', default=None, type=str, metavar='PATH',
                            help='resumes training from checkpoint from PATH')
    checkpoint.add_argument('--save-all', action='store_true', default=False,
                            help='saves checkpoint after every epoch')
    checkpoint.add_argument('--save-freq', default=5000, type=int,
                            help='save checkpoint every SAVE_FREQ batches')
    checkpoint.add_argument('--keep-checkpoints', default=0, type=int,
                            help='keep only last KEEP_CHECKPOINTS checkpoints, \
                        affects only checkpoints controlled by --save-freq \
                        option')

    # distributed support
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int,
                             help='number of processes, do not set! Done by multiproc module')
    distributed.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                             help='url used to set up distributed training')

    return parser.parse_args(argv)


def build_criterion(vocab_size, padding_idx, smoothing):
    if smoothing == 0.:
        logging.info(f'building CrossEntropyLoss')
        loss_weight = torch.ones(vocab_size)
        loss_weight[padding_idx] = 0
        criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
        mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_LOSS_FN,
                              value='Cross Entropy')
    else:
        logging.info(f'building SmoothingLoss (smoothing: {smoothing})')
        criterion = LabelSmoothing(padding_idx, smoothing)
        mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_LOSS_FN,
                              value='Cross Entropy with label smoothing')
        mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_LOSS_SMOOTHING,
                              value=smoothing)

    return criterion


def main(exp, argv):
    args = parse_args(argv)

    torch_settings = init_torch(
        seed=args.seed,
        workers=args.workers,
        cuda=args.cuda,
    )

    mlperf_log.ROOT_DIR_GNMT = os.path.dirname(os.path.abspath(__file__))
    mlperf_log.LOGGER.propagate = False
    mlperf_log.gnmt_print(key=mlperf_log.RUN_START)

    device = torch_settings.device

    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    # initialize distributed backend
    distributed = args.world_size > 1
    if distributed:
        backend = 'nccl' if args.cuda else 'gloo'
        dist.init_process_group(backend=backend, rank=args.rank,
                                init_method=args.dist_url,
                                world_size=args.world_size)

    # create directory for results
    save_path = exp.results_directory()

    # setup logging
    log_filename = f'log_gpu_{args.rank}.log'

    setup_logging(os.path.join(save_path, log_filename))

    if torch_settings.cuda:
        torch.cuda.set_device(args.rank)

    train_object = exp.get_dataset(
        args.dataset,
        config=config,
        min_length=args.min_length_train,
        max_length=args.max_length_train,
        max_size=args.max_size,
    )
    train_data = train_object.data
    tokenizer = train_object.tokenizer
    mlperf_log.gnmt_print(key=mlperf_log.PREPROC_NUM_TRAIN_EXAMPLES,
                          value=len(train_data))

    vocab_size = tokenizer.vocab_size
    mlperf_log.gnmt_print(key=mlperf_log.PREPROC_VOCAB_SIZE, value=vocab_size)

    # build GNMT model
    model_config = dict(vocab_size=vocab_size, math=args.math,
                        **literal_eval(args.model_config))
    model = models.GNMT(**model_config)
    logging.info(model)

    batch_first = model.batch_first

    # define loss function (criterion) and optimizer
    criterion = build_criterion(vocab_size, config.PAD, args.smoothing)
    opt_config = literal_eval(args.optimization_config)

    # create trainer
    trainer_options = dict(
        criterion=criterion,
        grad_clip=args.grad_clip,
        save_path=save_path,
        save_freq=args.save_freq,
        save_info={'config': args, 'tokenizer': tokenizer},
        opt_config=opt_config,
        batch_first=batch_first,
        keep_checkpoints=args.keep_checkpoints,
        math=args.math,
        print_freq=args.print_freq,
        cuda=args.cuda,
        distributed=distributed)

    trainer_options['model'] = model
    trainer = trainers.Seq2SeqTrainer(**trainer_options)

    translator = Translator(model,
                            tokenizer,
                            beam_size=args.beam_size,
                            max_seq_len=args.max_length_val,
                            len_norm_factor=args.len_norm_factor,
                            len_norm_const=args.len_norm_const,
                            cov_penalty_factor=args.cov_penalty_factor,
                            cuda=args.cuda)

    num_parameters = sum([l.nelement() for l in model.parameters()])

    # get data loaders
    train_loader = train_data.get_loader(batch_size=args.batch_size,
                                         batch_first=batch_first,
                                         shuffle=True,
                                         bucket=args.bucketing,
                                         num_workers=args.workers,
                                         drop_last=True,
                                         distributed=distributed)

    mlperf_log.gnmt_print(key=mlperf_log.INPUT_BATCH_SIZE,
                          value=args.batch_size * args.world_size)
    mlperf_log.gnmt_print(key=mlperf_log.INPUT_SIZE,
                          value=train_loader.sampler.num_samples)

    chrono = exp.chronos.create(
        "train",
        type="rate",
        sync=torch_settings.sync,
        sample_duration=args.sample_duration,
        max_count=args.max_count,
    )

    epoch = 0
    while not chrono.done():
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        trainer.epoch = epoch
        train_loss = trainer.optimize(train_loader, chrono)
        epoch += 1
