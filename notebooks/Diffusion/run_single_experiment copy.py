import torch
from torchvision.utils import make_grid, save_image
from diffusion_doodle_lib_temp import set_debug
set_debug(True)
from diffusion_doodle_lib_temp import step_end

from diffusion_doodle_lib_temp import (
    ModelConfig,
    build_model,
    get_schedule,
    train_step,
    sample,
    get_loader,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Using device:", device)

CFG = ModelConfig(
    T=1000,
    depth=3,
    base_channels=16,
    depth_per_stage=3,
)

model = build_model(CFG).to(device)
schedule = get_schedule(CFG).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

loader = get_loader(8)
data_iter = iter(loader)

print("Model ready. Starting training...\n")

for step in range(2000):
    try:
        batch = next(data_iter).to(device)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter).to(device)

    loss, grads, buckets = train_step(model, batch, schedule, opt)
    step_end(loss)
    if step % 50 == 0:
        print(step, loss)

print("\nSampling...")

imgs = sample(model, schedule, device, num_samples=16).clamp(-1, 1)
grid = make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))

save_image(grid, "single_experiment_samples.png")
print("saved")