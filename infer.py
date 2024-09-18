import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from gan_module import Generator

# Argument parser setup
parser = ArgumentParser()
parser.add_argument('--image_dir', default='/Users/jayanth/Face-Aging/Face-Aging/archive/UTKFace', help='The image directory')
parser.add_argument('--model_path', default='/Users/jayanth/Face-Aging/Face-Aging/pretrained_model/state_dict.pth', help='Path to the pretrained model file')

@torch.no_grad()
def main():
    print("Current working directory:", os.getcwd())
    args = parser.parse_args()
    
    # Check if image directory exists and is not empty
    if not os.path.exists(args.image_dir) or not os.listdir(args.image_dir):
        print(f"Directory not found or empty: {args.image_dir}")
        return
    
    # List images in directory
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if x.endswith(('.png', '.jpg'))]
    print("Image paths found:", image_paths)
    
    # Load model
    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()

    # Image transformations
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Determine number of images to display
    nr_images = min(len(image_paths), 6)
    fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
    random.shuffle(image_paths)
    
    for i in range(nr_images):
        img = Image.open(image_paths[i]).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)

    # Display or save the plot
    # plt.show()
    plt.savefig("aged_faces.png")

if __name__ == '__main__':
    main()
