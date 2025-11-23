from GS_DRUNet import UNetRes
import torch
from Trainer import Trainer
from hyperparameters import n_epochs
from hyperparameters import result_folder
# CUDA related hyperparameters
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(20)

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU : {}".format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    gs_std = UNetRes(in_nc=2, out_nc=2, nc=[64, 128, 256, 512], nb=3, act_mode='E', downsample_mode='strideconv', upsample_mode='convtranspose')
    trainer = Trainer(gs_std, device, pretrained_path=f"{result_folder}/drunet_gaussian/model_600.pth", experiment_name="drunet_gaussian", exp_type="u_v")

    trainer.train(n_epochs)
