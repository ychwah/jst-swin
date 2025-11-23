from GS_DRUNet import GradStepSingleChannel, GradStepSTD
import torch
import pandas as pd
from generate_std_images import generate_std
import numpy as np
from misc import save_img


noise_levels_255 = [0, 5, 10, 15, 20]
noise_std_levels = [i / 255 for i in noise_levels_255]

nb_images = 1000
structures = []


def psnr(img_1, img_2):
    max_value = 1.0
    if isinstance(img_1, np.ndarray):
        mse = np.mean((img_1 - img_2) ** 2)
    else:
        mse = torch.mean((img_1 - img_2) ** 2).item()
    if mse == 0:
        psnr_ = 100
    else:
        psnr_ = 10 * np.log10(max_value ** 2 / mse)
    return psnr_


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU : {}".format(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print("Running on the CPU")

if __name__ == "__main__":
    # R_u
    data_u = []
    model_u = GradStepSingleChannel()
    model_u.load_state_dict(torch.load("../models/R_u_gaussian/model_600.pth", map_location=device))
    model_u = model_u.to(device)

    data_v = []
    model_v = GradStepSingleChannel()
    model_v.load_state_dict(torch.load("../models/R_v_gaussian/model_600.pth", map_location=device))
    model_v = model_v.to(device)

    data_structure = []
    data_texture = []
    model = GradStepSTD()
    model.load_state_dict(torch.load("../models/R_u_v_gaussian/model_600.pth", map_location=device))
    model = model.to(device)

    for idx in range(nb_images):
        u_ref, v_ref = generate_std(64, 64)
        u_ref = torch.tensor(u_ref, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        v_ref = torch.tensor(v_ref, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        std_ref = torch.concatenate([u_ref, v_ref])

        u_ref = u_ref.to(device)
        v_ref = v_ref.to(device)
        std_ref = std_ref.to(device)

        noises_u = [torch.normal(mean=torch.zeros((64, 64)), std=sigma).to(device) for sigma in noise_std_levels]
        noises_v = [torch.normal(mean=torch.zeros((64, 64)), std=sigma).to(device) for sigma in noise_std_levels]

        psnr_u = {"image_id": idx, **{noise_level: 0 for noise_level in noise_levels_255}}
        psnr_v = {"image_id": idx, **{noise_level: 0 for noise_level in noise_levels_255}}
        psnr_structure = {"image_id": idx, **{noise_level: 0 for noise_level in noise_levels_255}}
        psnr_texture = {"image_id": idx, **{noise_level: 0 for noise_level in noise_levels_255}}

        output_exp = []
        for noise_amp, noise_s, noise_t in zip(noise_levels_255, noises_u, noises_v):
            u_noisy = u_ref + noise_s
            v_noisy = v_ref + noise_t
            std_noisy = torch.concatenate([u_noisy, v_noisy], 1).to(device)

            # R_u
            out_u = model_u(u_noisy).detach()
            psnr_u[noise_amp] = psnr(out_u, u_ref)

            # R_v
            out_v = model_v(v_noisy).detach()
            psnr_v[noise_amp] = psnr(out_v, v_ref)

            # R_u_v
            out = model(std_noisy).detach()
            psnr_structure[noise_amp] = psnr(out[:, :1, ...], u_ref)
            psnr_texture[noise_amp] = psnr(out[:, 1:, ...], v_ref)

            if idx % 50 == 0:
                output_img = np.ones((64+5, 64*6 + 5 * 5))
                output_img[:64, :64] = u_noisy[0, 0, ...].detach().cpu().numpy()
                output_img[:64, 64 + 5: 2*64 + 5] = out[0, 0, ...].cpu().numpy()
                output_img[:64, 2*64 + 10:3*64 + 10] = out_u[0, 0, ...].cpu().numpy()
                output_img[:64, 3*64+15:4*64+15] = v_noisy[0, 0, ...].detach().cpu().numpy() + 0.5
                output_img[:64, 4*64+20:5*64+20] = out[0, 1, ...].cpu().numpy() + 0.5
                output_img[:64, 5 * 64 + 25: 6*64 + 25] = out_v[0, 0, ...].cpu().numpy() + 0.5
                output_exp.append(output_img)

        if idx % 100 == 0:
            save_img(np.concatenate(output_exp, 0), f"../test_denoising/{idx}.png")
        data_u.append(psnr_u)
        data_v.append(psnr_v)
        data_structure.append(psnr_structure)
        data_texture.append(psnr_texture)

    # save csv
    df = pd.DataFrame(data_u)
    df.to_csv("../test_denoising/structure/R_u.csv")
    df = pd.DataFrame(data_structure)
    df.to_csv("../test_denoising/structure/R_u_v.csv")

    df = pd.DataFrame(data_v)
    df.to_csv("../test_denoising/texture/R_v.csv")
    df = pd.DataFrame(data_texture)
    df.to_csv("../test_denoising/texture/R_u_v.csv")
