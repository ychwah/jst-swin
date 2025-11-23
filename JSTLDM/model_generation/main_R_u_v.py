from GS_DRUNet import GradStepSTD
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

    gs_std = GradStepSTD()
    trainer = Trainer(gs_std, device, pretrained_path=None, experiment_name="R_u_v_limited", exp_type="u_v")

    trainer.train(n_epochs)
