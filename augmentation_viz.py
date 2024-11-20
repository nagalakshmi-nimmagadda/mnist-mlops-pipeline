import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

def show_augmented_images(num_samples=5):
    # Define augmentation pipeline without initial ToTensor
    augmentation_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])

    # Load MNIST dataset with just ToTensor for original images
    base_transform = transforms.ToTensor()
    dataset = MNIST('./data', train=True, download=True, transform=base_transform)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))
    
    for i in range(num_samples):
        # Get original image
        orig_img, label = dataset[i]
        
        # Generate 4 augmented versions
        augmented = []
        for _ in range(4):
            # Convert tensor to PIL Image for augmentation
            img_pil = transforms.ToPILImage()(orig_img)
            # Apply augmentations
            aug_img = augmentation_transform(img_pil)
            # Convert back to tensor
            aug_tensor = base_transform(aug_img)
            augmented.append(aug_tensor)
        
        # Plot original and augmented images
        imgs = [orig_img] + augmented
        for j, img in enumerate(imgs):
            ax = axes[i, j]
            img_np = img.squeeze().numpy()
            ax.imshow(img_np, cmap='gray')
            ax.axis('off')
            if j == 0:
                ax.set_title(f'Original\nLabel: {label}')
            else:
                ax.set_title(f'Augmented {j}')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png')
    plt.close()

if __name__ == "__main__":
    show_augmented_images() 