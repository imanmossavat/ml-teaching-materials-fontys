import torch
import matplotlib.pyplot as plt

# path from your code
cache_path = "data/mnist_64/mnist64_train.pt"

payload = torch.load(cache_path, map_location="cpu")
images = payload["images"]   # (N, 1, 64, 64)

print("Shape:", images.shape)
print("Min:", images.min().item())
print("Max:", images.max().item())

def show_grid(images, n=16):
    # take first n*n images
    imgs = images[:n*n]

    # convert [-1,1] → [0,1] for display
    imgs = (imgs + 1) / 2.0

    fig, axes = plt.subplots(n, n, figsize=(6, 6))

    k = 0
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(imgs[k, 0], cmap="gray")
            ax.axis("off")
            k += 1

    plt.tight_layout()
    plt.show()

show_grid(images, n=6)