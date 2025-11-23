import torch
from .GS_DRUNet import GradStepSTD
from .patch_extraction import PatchOperator
import numpy as np
model_path_R_u_v = "./saved_models/R_u_v.pth"


def get_projection(f):
    def projection(x):
        tmp = x[:, :1, :, :] - x[:, 1:, :, :]
        return 0.5 * torch.concatenate([f + tmp, f - tmp], 1)

    return projection

def joint_decomposition_synthetic(input_img, nb_iter, mu, device):
    f_ref = input_img

    # load model
    model = GradStepSTD()
    model.load_state_dict(torch.load(model_path_R_u_v, map_location=device))
    model = model.to(device)

    # setup projection
    proj_cf = get_projection(torch.tensor(f_ref, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))

    # initialise the algorithm
    u_init = torch.tensor(f_ref, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    v_init = torch.zeros_like(u_init)

    std = torch.concatenate([u_init, v_init], 1).to(device)
    for iter_ in range(nb_iter):
        z = (1 - mu) * std + mu * model(std).detach()
        std = proj_cf(z)

    u_out = std[0, 0, :, :].detach().cpu().numpy()
    v_out = std[0, 1, :, :].detach().cpu().numpy()

    return u_out, v_out

def joint_decomposition_natural_images(input_img, nb_iter, mu, device):
    f_ref = input_img
    f_tensor = torch.tensor(f_ref, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    proj_cf = get_projection(torch.tensor(f_ref, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))


    model = GradStepSTD()
    model.load_state_dict(torch.load(model_path_R_u_v, map_location=device))
    model = model.to(device)



    window_size = (64, 64)
    stride = (48, 48)

    b, c, n, m = f_tensor.shape
    po = PatchOperator(original_size=(n, m), window_size=window_size, stride=stride)
    po.normalization_map = po.normalization_map.to(device)
    f_patch = po.patchify(f_tensor)

    nb_patches = f_patch.shape[0]
    std = torch.tensor([f_ref, np.zeros_like(f_ref)], dtype=torch.float32).unsqueeze(0)
    std = std.to(device)
    
    if device.type == "cpu":
        nb_of_patches_batch = 4  # set this up depending on your available compute power
    else:
        nb_of_patches_batch = 16
    for iter_ in range(nb_iter):
        std_patch = po.patchify(std)
        for p_idx in range(0, nb_patches, nb_of_patches_batch):
            curr_patch = std_patch[p_idx:p_idx + nb_of_patches_batch]
            std_patch[p_idx:p_idx + nb_of_patches_batch] = (1 - mu) * curr_patch + mu * model(curr_patch).detach()
        z = po.recover_from_patch(std_patch)
        std = proj_cf(z)

    u_out = std[0, 0, :, :].detach().cpu().numpy()
    v_out = std[0, 1, :, :].detach().cpu().numpy()

    return u_out, v_out

def joint_decomposition(input_img, nb_iter, mu, device, gpu_compute):

    if max(input_img.shape) > 64:
        return joint_decomposition_natural_images(input_img, nb_iter, mu, device)
    else:
        return joint_decomposition_synthetic(input_img, nb_iter, mu, device)
