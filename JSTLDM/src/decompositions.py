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
    f_tensor = torch.tensor(f_ref, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # ----------------------------
    # FIX: PAD IMAGE FOR FULL COVERAGE
    # ----------------------------
    n, m = f_tensor.shape[-2], f_tensor.shape[-1]
    window_size = (64, 64)
    stride = (48, 48)

    # compute needed padding
    pad_h = (window_size[0] - (n % stride[0])) % stride[0]
    pad_w = (window_size[1] - (m % stride[1])) % stride[1]

    # apply padding (reflect padding avoids edge artifacts)
    f_tensor_padded = torch.nn.functional.pad(
        f_tensor, (0, pad_w, 0, pad_h), mode="reflect"
    )

    padded_n, padded_m = f_tensor_padded.shape[-2], f_tensor_padded.shape[-1]

    # projection operator uses the padded reference
    proj_cf = get_projection(f_tensor_padded.clone())

    # load model
    model = GradStepSTD()
    model.load_state_dict(torch.load(model_path_R_u_v, map_location=device))
    model = model.to(device)

    # patch operator built on padded size
    po = PatchOperator(
        original_size=(padded_n, padded_m),
        window_size=window_size,
        stride=stride,
    )
    po.normalization_map = po.normalization_map.to(device)

    # initial std (u,v)
    stacked = np.stack([f_ref, np.zeros_like(f_ref)], axis=0).astype(np.float32)
    std = torch.from_numpy(stacked).unsqueeze(0).to(device)

    # >>> IMPORTANT: pad std as well so it matches f_tensor_padded <<<
    std = torch.nn.functional.pad(std, (0, pad_w, 0, pad_h), mode="reflect")

    # patchify reference also
    f_patch = po.patchify(f_tensor_padded)

    nb_patches = f_patch.shape[0]

    # batch size for patch processing
    if device.type == "cpu":
        nb_of_patches_batch = 4
    else:
        nb_of_patches_batch = 16

    # main iterations
    for iter_ in range(nb_iter):

        std_patch = po.patchify(std)

        # batch processing of patches
        for p_idx in range(0, nb_patches, nb_of_patches_batch):
            batch = std_patch[p_idx:p_idx + nb_of_patches_batch]
            std_patch[p_idx:p_idx + nb_of_patches_batch] = \
                (1 - mu) * batch + mu * model(batch).detach()

        # reconstruct from patches
        z = po.recover_from_patch(std_patch)

        # projection
        std = proj_cf(z)

    # remove padding → back to original image size
    std = std[..., :n, :m]

    # extract outputs
    u_out = std[0, 0].detach().cpu().numpy()
    v_out = std[0, 1].detach().cpu().numpy()

    return u_out, v_out

def joint_decomposition(input_img, nb_iter, mu, device, gpu_compute):

    if max(input_img.shape) > 64:
        return joint_decomposition_natural_images(input_img, nb_iter, mu, device)
    else:
        return joint_decomposition_synthetic(input_img, nb_iter, mu, device)
