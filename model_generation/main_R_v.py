from GS_DRUNet import GradStepSingleChannel
import torch
from Trainer import Trainer
from hyperparameters import n_epochs
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

    gs_std = GradStepSingleChannel()
    trainer = Trainer(gs_std, device, pretrained_path="../models/R_v_gaussian/model_LAST.pth", experiment_name="R_v_gaussian", exp_type="v")

    trainer.train(n_epochs)
