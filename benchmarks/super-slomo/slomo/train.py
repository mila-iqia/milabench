# [Super SloMo]
##High Quality Estimation of Multiple Intermediate Frames for Video Interpolation

import argparse
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
from giving import give
import voir
from synth import SyntheticData


def has_xpu():
    try:
        import intel_extension_for_pytorch as ipex
        return torch.xpu.is_available()
    except ImportError as err:
        return True
    

device_interface = None
backend_optimizer = lambda x, y, **kwargs: (x, y)
device_name = "cpu"
if has_xpu():
    device_name = "xpu"
    device_interface = torch.xpu
    backend_optimizer = device_interface.optimize

if torch.cuda.is_available():
    device_name = "cuda"
    device_interface = torch.cuda



def main():
    # For parsing commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=False,
        help="path to dataset folder containing train-test-validation folders",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="path of checkpoint for pretrained model"
    )
    parser.add_argument(
        "--train_continue",
        type=bool,
        default=False,
        help="If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="number of epochs to train. Default: 200.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=6,
        help="batch size for training. Default: 6.",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=10,
        help="batch size for validation. Default: 10.",
    )
    parser.add_argument(
        "--init_learning_rate",
        type=float,
        default=0.0001,
        help="set initial learning rate. Default: 0.0001.",
    )
    parser.add_argument(
        "--milestones",
        type=list,
        default=[100, 150],
        help="Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]",
    )
    parser.add_argument(
        "--progress_iter",
        type=int,
        default=100,
        help="frequency of reporting progress and validation. N: after every N iterations. Default: 100.",
    )
    parser.add_argument(
        "--no-tf32",
        dest="allow_tf32",
        action="store_false",
        help="do not allow tf32",
    )

    args = parser.parse_args()

    if args.allow_tf32:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    if torch.xpu.is_available():
        import intel_extension_for_pytorch as ipex
        if args.allow_tf32:
            ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.TF32)
        else:
            ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.FP32)


    ###Initialize flow computation and arbitrary-time flow interpolation CNNs.

    device = torch.device(f"{device_name}:0")
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)

    ###Initialze backward warpers for train and validation datasets

    trainFlowBackWarp = model.backWarp(352, 352, device)
    trainFlowBackWarp = trainFlowBackWarp.to(device)
    validationFlowBackWarp = model.backWarp(640, 352, device)
    validationFlowBackWarp = validationFlowBackWarp.to(device)

    ###Load Datasets

    # # Channel wise mean calculated on adobe240-fps training dataset
    # mean = [0.429, 0.431, 0.397]
    # std  = [1, 1, 1]
    # normalize = transforms.Normalize(mean=mean,
    #                                 std=std)
    # transform = transforms.Compose([transforms.ToTensor(), normalize])

    # trainset = dataloader.SuperSloMo(root=args.dataset_root + '/train', transform=transform, train=True)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=False)

    def igen():
        sz = 352
        f0 = torch.rand((3, sz, sz)) * 2 - 1
        ft = torch.rand((3, sz, sz)) * 2 - 1
        f1 = torch.rand((3, sz, sz)) * 2 - 1
        return [f0, ft, f1]

    def ogen():
        return torch.randint(0, 7, ())

    trainset = SyntheticData(
        n=args.train_batch_size, repeat=10000, generators=[igen, ogen]
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, num_workers=2
    )

    ###Utils

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    ###Loss and Optimizer

    L1_lossFn = nn.L1Loss()
    MSE_LossFn = nn.MSELoss()

    params = list(ArbTimeFlowIntrp.parameters()) + list(flowComp.parameters())

    optimizer = optim.Adam(params, lr=args.init_learning_rate)

    # scheduler to decrease learning rate by a factor of 10 at milestones.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1
    )

    ###Initializing VGG16 model for perceptual loss

    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
    vgg16_conv_4_3.to(device)
    vgg16_conv_4_3.eval()
    for param in vgg16_conv_4_3.parameters():
        param.requires_grad = False


    ArbTimeFlowIntrp, optimizer = backend_optimizer(ArbTimeFlowIntrp, optimizer=optimizer, dtype=torch.float)

    flowComp, optimizer = backend_optimizer(flowComp, optimizer=optimizer, dtype=torch.float)

    vgg16 = backend_optimizer(vgg16_conv_4_3, optimizer=None, dtype=torch.float)

    ### Initialization

    if args.train_continue:
        dict1 = torch.load(args.checkpoint)
        ArbTimeFlowIntrp.load_state_dict(dict1["state_dictAT"])
        flowComp.load_state_dict(dict1["state_dictFC"])
    else:
        dict1 = {"loss": [], "valLoss": [], "valPSNR": [], "epoch": -1}

    ### Training

    cLoss = dict1["loss"]
    valLoss = dict1["valLoss"]
    valPSNR = dict1["valPSNR"]

    ### Main training loop
    for epoch in range(dict1["epoch"] + 1, args.epochs):
        print("Epoch: ", epoch)

        # Append and reset
        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        iLoss = 0

        # Increment scheduler count
        scheduler.step()

        # for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):
        for trainIndex, (trainData, trainFrameIndex) in enumerate(
            voir.iterate(
                "train",
                trainloader,
                report_batch=True,
                batch_size=lambda batch: batch[1].shape[0],
            ),
            0,
        ):
            ## Getting the input and the target from the training set
            frame0, frameT, frame1 = trainData

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)

            optimizer.zero_grad()

            # Calculate flow between reference frames I0 and I1
            flowOut = flowComp(torch.cat((I0, I1), dim=1))

            # Extracting flows between I0 and I1 - F_0_1 and F_1_0
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            fCoeff = model.getFlowCoeff(trainFrameIndex, device)

            # Calculate intermediate flows
            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            # Get intermediate frames from the intermediate flows
            g_I0_F_t_0 = trainFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = trainFlowBackWarp(I1, F_t_1)

            # Calculate optical flow residuals and visibility maps
            intrpOut = ArbTimeFlowIntrp(
                torch.cat(
                    (I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1
                )
            )

            # Extract optical flow residuals and visibility maps
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0

            # Get intermediate frames from the intermediate flows
            g_I0_F_t_0_f = trainFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = trainFlowBackWarp(I1, F_t_1_f)

            wCoeff = model.getWarpCoeff(trainFrameIndex, device)

            # Calculate final intermediate frame
            Ft_p = (
                wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f
            ) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            # Loss
            recnLoss = L1_lossFn(Ft_p, IFrame)

            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))

            warpLoss = (
                L1_lossFn(g_I0_F_t_0, IFrame)
                + L1_lossFn(g_I1_F_t_1, IFrame)
                + L1_lossFn(trainFlowBackWarp(I0, F_1_0), I1)
                + L1_lossFn(trainFlowBackWarp(I1, F_0_1), I0)
            )

            loss_smooth_1_0 = torch.mean(
                torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])
            ) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(
                torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])
            ) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

            # Total Loss - Coefficients 204 and 102 are used instead of 0.8 and 0.4
            # since the loss in paper is calculated for input pixels in range 0-255
            # and the input to our network is in range 0-1
            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
            give(loss=loss.item())
            # Backpropagate
            loss.backward()
            optimizer.step()
            iLoss += loss.item()


if __name__ == "__main__":
    main()
