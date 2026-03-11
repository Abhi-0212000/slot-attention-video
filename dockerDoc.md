
### 3. How to "get in"

Clone the repo and enter the directory:

```bash
git clone https://github.com/Abhi-0212000/slot-attention-video.git
cd slot-attention-video
git checkout cuda12-compat
```

**Step A: Build the image (you only do this once)**

```bash
docker compose build

```

*(This will take a few minutes as it installs Python, JAX 0.4.13 with CUDA 12 GPU wheels, and other dependencies.)*

**Step B: Start and enter the container**

```bash
docker compose run --rm savi

```

This command fires up the container, connects your terminal to it, and drops you straight into the `/workspace/slot-attention-video` directory as the `root` user. The `--rm` flag just means the container cleans up after itself when you type `exit`.

### A quick test once you are inside:

Once your terminal prompt changes to show you are inside the Docker container, run this to guarantee the container sees your GPU:

```bash
python -c "import jax; print(jax.devices())"

```

If it outputs `[CudaDevice(id=0)]`, you are fully set up and ready to run the SAVi training scripts!