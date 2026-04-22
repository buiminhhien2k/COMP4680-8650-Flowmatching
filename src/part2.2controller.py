import itertools
import subprocess
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

datasets = ["swiss_roll", "gaussians", "circles"]
dims = [2, 8, 32]
preds = ["x", "v"]
losses = ["x", "v"]

for dataset, dim, pred, loss in itertools.product(datasets, dims, preds, losses):
    cmd = [
        "python", "./src/part2.2.py",
        "--dataset", dataset,
        "--dim", str(dim),
        "--pred-type", pred,
        "--loss-type", loss,
        # training hyperparameters
        "--print-every", str(100),
        "--train-steps", str(25000),
        "--eps-clip", str(0.05)
    ]
    subprocess.run(cmd, check=True)