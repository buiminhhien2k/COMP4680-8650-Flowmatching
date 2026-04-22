from dataloader import get_dataloader

import matplotlib.pyplot as plt

AVAILABLE_DATASETS = ("swiss_roll", "gaussians", "circles")
AVAILABLE_DIMS = (2, 32)

r, c = len(AVAILABLE_DATASETS), len(AVAILABLE_DIMS)
fig, ax = plt.subplots(nrows=r, ncols=c, figsize=(c * 2.5, r * 2.5))

for i in range(r):
    for j in range(c):
        dataset = AVAILABLE_DATASETS[i]
        dim = AVAILABLE_DIMS[j]
        data_loader2d = get_dataloader(dataset, dim=dim, shuffle=True)
        to_2d_func = data_loader2d.dataset.to_2d

        for batch in data_loader2d:
            if dim != 2:
                batch = to_2d_func(batch)
            x, y = batch[:, 0], batch[:, 1]
            ax[i, j].scatter(x, y, s=0.4)
            ax[i, j].set_title(f"{dataset}-{dim}d")

plt.tight_layout()
plt.show()