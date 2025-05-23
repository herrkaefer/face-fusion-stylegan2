# -*- coding: utf-8 -*-

import os
import sys
import subprocess

# Clone required repositories if they don't exist
def clone_repository(repo_url, target_dir):
    if not os.path.isdir(target_dir):
        print(f"Cloning {target_dir} repository...")
        try:
            subprocess.run(['git', 'clone', repo_url, target_dir], check=True)
            print(f"{target_dir} repository cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning {target_dir}: {e}")
            raise

# Clone required repositories
clone_repository('https://github.com/omertov/encoder4editing.git', './encoder4editing')
clone_repository('https://github.com/NVlabs/stylegan2-ada-pytorch.git', './stylegan2-ada-pytorch')

sys.path.append('./stylegan2-ada-pytorch')
sys.path.append('./encoder4editing')

import numpy as np
import PIL.Image
import cv2
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import lpips
from tqdm import tqdm
import mediapipe as mp
from face_alignment import FaceAlignment, LandmarksType
# import matplotlib.pyplot as plt
# import math
import requests
import os
import gdown
import bz2
import shutil

import legacy, dnnlib
from projector import project

from argparse import Namespace
from models.psp import pSp
import dlib
from utils.alignment import align_face

project_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_dir, 'assets')

stylegan_model_path = os.path.join(model_dir, 'ffhq.pkl')
e4e_model_path = os.path.join(model_dir, 'e4e_ffhq_encode.pt')
shape_predictor_path = os.path.join(model_dir, 'shape_predictor_68_face_landmarks.dat')

os.makedirs(model_dir, exist_ok=True)

if not os.path.isfile(stylegan_model_path):
    ada_stylegan_model_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
    print(f"Downloading StyleGAN2 model from {ada_stylegan_model_url}...")
    response = requests.get(ada_stylegan_model_url)
    with open(stylegan_model_path, 'wb') as f:
        f.write(response.content)
    print(f"ADA StyleGAN2 pretrained model downloaded and saved to {stylegan_model_path}")

if not os.path.isfile(e4e_model_path):
    e4e_model_url = 'https://drive.google.com/uc?id=1cUv_reLE6k3604or78EranS7XzuVMWeO'
    print(f"Downloading e4e model from {e4e_model_url}...")
    gdown.download(e4e_model_url, e4e_model_path, quiet=False)
    print(f"e4e model downloaded and saved to {e4e_model_path}")

if not os.path.isfile(shape_predictor_path):
    shape_predictor_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    print(f"Downloading shape predictor model from {shape_predictor_url}...")
    response = requests.get(shape_predictor_url)
    bz2_path = shape_predictor_path + '.bz2'

    # Save the compressed file
    with open(bz2_path, 'wb') as f:
        f.write(response.content)

    # Decompress the file
    print("Decompressing the shape predictor model...")
    with bz2.BZ2File(bz2_path, 'rb') as source, open(shape_predictor_path, 'wb') as dest:
        dest.write(source.read())

    # Remove the compressed file
    os.remove(bz2_path)
    print(f"Shape predictor model downloaded and decompressed to {shape_predictor_path}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

G = None # StyleGAN2 generator
e4e_net = None # e4e encoder


# Function to load pretrained generator
def load_stylegan_generator(network_pkl, device=device):
    global G
    if G is None:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # Only the generator is needed
    return G


def tensor_to_PIL(img_tensor):
    if img_tensor.ndim == 3:
      img_tensor = img_tensor.unsqueeze(0)
    # img is of shape [1, 3, H, W], in range [-1, 1]
    img = (img_tensor + 1) * (255 / 2)
    img = img.clamp(0, 255).to(torch.uint8)
    img = img[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    return PIL.Image.fromarray(img, 'RGB')


def PIL_to_sensor(pil_image, target_size=1024):
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),  # Converts to [0, 1]
    ])
    # Convert [0,1] to [0,255] range
    img = transform(pil_image) * 255.0
    return img  # Shape: [3, H, W] in range [0, 255]


# Function to generate an image from a random latent vector
def generate_random_image(G, device=device, truncation_psi=0.7, outdir='outputs', filename='gen_img.jpg'):
    z = torch.randn([1, G.z_dim], device=device)
    print(f"{z.shape=}")
    label = torch.zeros([1, G.c_dim], device=device)
    image_tensor = G(z, label, truncation_psi=truncation_psi, noise_mode='const')

    print(f"img shape: {image_tensor.shape}")

    # Convert to PIL image and save
    pil_img = tensor_to_PIL(image_tensor)
    print(f"pil_img shape: {pil_img.size}")

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, filename)
    pil_img.save(out_path)
    print(f"Image saved to {out_path}")
    return image_tensor, pil_img


def gan_inversion(G, pil_image, num_steps=100, reconstruct=False):
    """
    Projects a PIL image into W latent space using StyleGAN2-ADA projector.

    Returns:
        - projected_w: Tensor [1, 1, 512]
        - projected_img: Tensor [1, 3, 256, 256]
    """
    # Set eval mode
    G.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = PIL_to_sensor(pil_image)

    print(f"target shape: {target.shape}")
    print(f"{(G.img_channels, G.img_resolution, G.img_resolution)=}")

    projected_w = project(
        G=G,
        target=target,
        num_steps=num_steps,
        initial_learning_rate=0.01,  # Lower initial learning rate
        # lr_rampdown_length=0.75,     # Slower learning rate decay
        # lr_rampup_length=0.25,       # Slower learning rate warmup
        # noise_ramp_length=0.9,       # Longer noise schedule
        regularize_noise_weight=1e4,  # Lower regularization
        device=device,
        verbose=True
    )

    print(f"projected_w shape: {projected_w.shape}")

    final_w = projected_w[-1]

    # 生成重建图像
    if reconstruct:

      # final_w = projected_w[-1:].to(device)  # shape [1, num_ws, w_dim]
      # print(f"final_w shape: {final_w.shape}")
      # projected_img = G.synthesis(final_w, noise_mode='const')

      # label = torch.zeros([1, G.c_dim], device=device)
      # projected_img = G(final_w.to(device), label, noise_mode='const')

      print(f"{final_w.unsqueeze(0).shape=}")
      projected_img = G.synthesis(final_w.unsqueeze(0), noise_mode='const')

    else:
      projected_img = None

    return final_w, projected_img


# --------------------------------------


def load_e4e_encoder():
    global e4e_net
    if e4e_net is not None:
        return e4e_net

    print(f"Loading e4e encoder from {e4e_model_path}")

    try:
        # load the e4e model
        ckpt = torch.load(e4e_model_path, map_location=torch.device('cuda'))
        print(f"{ckpt.keys()=}")
        opts = ckpt['opts']
        opts['checkpoint_path'] = e4e_model_path
        opts= Namespace(**opts)
        print(f"{opts=}")

        # The checkpoint contains 'state_dict' and 'opts' instead of 'e4e'

        e4e_net = pSp(opts)
        e4e_net.load_state_dict(ckpt['state_dict'])

    except Exception as e:
        print(f"Error loading e4e: {e}")
        return None, None

    # Move model to GPU and set to eval mode
    e4e_net.cuda().eval()
    print("e4e encoder loaded.")
    return e4e_net


def e4e_inversion(e4e_encoder, pil_image, reconstruct=False):
    """
    Projects a PIL image into W latent space using e4e encoder.
    Optimized for Colab GPU usage.

    Args:
        G: StyleGAN2 generator
        pil_image: Input PIL image
        reconstruct: Whether to generate the reconstructed image

    Returns:
        - latent_codes: Tensor [1, 18, 512] (W+ space)
        - reconstructed_image: Tensor [1, 3, H, W] if reconstruct=True, else None
    """

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Move image to GPU
    img_tensor = transform(pil_image).unsqueeze(0).to("cuda").float()

    try:
        # Get latent codes with GPU acceleration
        with torch.no_grad():
            images, latents = e4e_encoder(img_tensor, randomize_noise=False, return_latents=True)
            # result = e4e_encoder(img_tensor)
            print(f"{len(images)=}")
            print(f"{latents.shape=}")
            latent = latents[0]  # Get W+ latent codes
            reconstructed_image = images[0]
            print(f"{latent.shape=}, {reconstructed_image.shape=}")

        return latent, reconstructed_image

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU out of memory. Clearing cache and trying again...")
            torch.cuda.empty_cache()
            return e4e_inversion(G, pil_image, reconstruct)  # Recursive retry
        else:
            print(f"Error during inference: {e}")
            return None, None


def image_from_latent(G, latent):
    with torch.no_grad():
        img = G.synthesis(
            latent.unsqueeze(0),
            noise_mode='const'
        )
    return img


def run_alignment(image_path):
    shape_predictor_path = os.path.join(model_dir, 'shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(shape_predictor_path)
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def fuse_latents(latent1, latent2, random_seed=None, variation_strength=0.0):
    """
    Fuses two latent codes to create a new latent that represents a mix of both inputs.

    Args:
        latent1: First parent's latent code tensor [18, 512]
        latent2: Second parent's latent code tensor [18, 512]
        random_seed: Seed for reproducibility (optional)
        variation_strength: Controls how much random variation to add (0-1)

    Returns:
        Tensor [18, 512]: A new latent code representing the "child"
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Ensure inputs are on the same device
    device = latent1.device

    # Generate random weights for mixing
    # We'll use different weights for different layers to allow for more natural mixing
    # mix_weights = torch.rand(latent1.shape[0], 1, device=device)  # [18, 1]
    # mix_weights = torch.full((latent1.shape[0], 1), 0.5, device=device)
    mix_weights = 0.2 + 0.6 * torch.rand(latent1.shape[0], 1, device=device)
    # mix_weights = torch.randint(0, 2, (latent1.shape[0], 1), device=device).float()

    # Create the base mixed latent
    mixed_latent = latent1 * mix_weights + latent2 * (1 - mix_weights)

    # Add some random variation to simulate genetic diversity
    if variation_strength > 0:
        variation = torch.randn_like(mixed_latent) * variation_strength
        mixed_latent = mixed_latent + variation

    return mixed_latent


def load_interfacegan_directions():
    """
    Loads the InterFaceGAN direction vectors for editing
    """
    directions = {
        'age': 'encoder4editing/editings/interfacegan_directions/age.pt',
        'smile': 'encoder4editing/editings/interfacegan_directions/smile.pt',
        'pose': 'encoder4editing/editings/interfacegan_directions/pose.pt'
    }

    loaded_directions = {}
    for name, path in directions.items():
        if os.path.exists(path):
            loaded_directions[name] = torch.load(path).cuda()
            print(f"Loaded {name} direction")

    if not loaded_directions:
        print("Warning: No InterFaceGAN directions found")

    return loaded_directions


def fuse_faces(input_img1, input_img2, random_seed=None, age_factor=0, smile_factor=0, variation_strength=0):
    """
    Test function to demonstrate face fusion between two parent images
    """

    G = load_stylegan_generator(stylegan_model_path)
    G.cuda().eval()

    e4e_encoder = load_e4e_encoder()

    # Align both input faces
    aligned_image1 = run_alignment(input_img1)
    aligned_image2 = run_alignment(input_img2)

    # Convert to tensors and move to GPU
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img_tensor1 = transform(aligned_image1).unsqueeze(0).cuda()
    img_tensor2 = transform(aligned_image2).unsqueeze(0).cuda()

    # Get latent codes for both parents
    with torch.no_grad():
        _, latent1 = e4e_encoder(img_tensor1, randomize_noise=False, return_latents=True)
        _, latent2 = e4e_encoder(img_tensor2, randomize_noise=False, return_latents=True)

        # Get the W+ latent codes
        latent1 = latent1[0]
        latent2 = latent2[0]

    # Generate child latent
    child_latent = fuse_latents(
        latent1,
        latent2,
        random_seed=random_seed,
        variation_strength=variation_strength
    )

    # Apply age modification if direction is available
    directions = load_interfacegan_directions()
    if 'age' in directions and age_factor != 0:
        child_latent += directions['age'] * age_factor

    if 'smile' in directions and smile_factor != 0:
        child_latent += directions['smile'] * smile_factor

    child_tensor = image_from_latent(G, child_latent)
    child_pil = tensor_to_PIL(child_tensor)

    del G, img_tensor1, img_tensor2, latent1, latent2, child_tensor
    torch.cuda.empty_cache()

    return child_pil


# Example usage
def test_e4e_inversion(e4e_encoder, input_img):
    """
    Test function to demonstrate e4e inversion
    """
    # Load StyleGAN2 generator
    G = load_stylegan_generator(model_path)
    G.cuda().eval()

    input_image = run_alignment(input_img)

    # Perform inversion
    latent, result_image = e4e_inversion(e4e_encoder, input_image, reconstruct=True)
    # print(f"{type(result_image)=}, {result_image.min()=}, {result_image.max()=}")

    if result_image is not None:
        # Save result image
        pil_result = tensor_to_PIL(result_image)
        os.makedirs('outputs', exist_ok=True)
        pil_result.save('outputs/e4e_result.jpg')
        print("Result image saved to outputs/e4e_result.jpg")

        # Reconstruct image from latent
        reconstruct_tensor = image_from_latent(G, latent)

        # reconstruct_tensor, _ = e4e_encoder.decoder(
        #     [latent],
        #     input_is_latent=True,
        #     randomize_noise=False,
        #     return_latents=True
        # )
        pil_reconstruct = tensor_to_PIL(reconstruct_tensor)
        pil_reconstruct.save('outputs/e4e_reconstruct.jpg')
        print("Reconstructed image saved to outputs/e4e_reconstruct.jpg")

    del G, latent, result_image
    torch.cuda.empty_cache()


def test_gan_inversion(input_img):
    # 1. 加载模型
    G = load_stylegan_generator(stylegan_model_path)

    # 2. 加载图片
    pil_image = PIL.Image.open(input_img).convert('RGB')

    # 3. inversion
    w, img_recon = gan_inversion(G, pil_image, num_steps=200, reconstruct=True)

    print(f"img_recon shape: {img_recon.shape}")

    # 4. 保存
    pil_img = tensor_to_PIL(img_recon)
    print(f"pil_img shape: {pil_img.size}")
    pil_img.save('outputs/reconstructed.jpg')
    print(f"Image saved to outputs/reconstructed.jpg")

