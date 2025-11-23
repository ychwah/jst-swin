import torch
from src import create_folder, get_img, joint_decomposition, save_img
import numpy as np
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU : {}".format(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print("Running on the CPU")

if __name__ == "__main__":
    output_folder = "./results"
    create_folder(output_folder)

    input_img = get_img("./dataset/test_dataset_synthetic/image_8.png")
    # input_img = get_img("./dataset/natural_images/Barbara.png")

    # structure/texture decomposition using a single function regularisation function R(u,v)
    u, v = joint_decomposition(input_img, nb_iter=10, mu=1.0, device=device)

    # save image
    n,m = input_img.shape
    output_img = np.ones((n, 3*m + 20))
    output_img[:, :m] = u
    output_img[:, m+10:2*m + 10] = v + 0.5
    output_img[:, 2*m + 20:] = u + v
    save_img(output_img, f"{output_folder}/Barbarba_R_u_v.png")
