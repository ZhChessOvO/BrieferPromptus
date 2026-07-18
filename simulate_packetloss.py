"""
simulate_packetloss.py — Simulate packet loss during prompt transmission using
the Gilbert-Elliott (GE) model.

Evaluates the quality (PSNR / SSIM / LPIPS) of generated frames when
higher-order prompt components (beyond min_rank) are randomly lost due to
packet loss, mimicking what would happen under unreliable network conditions.

The GE model alternates between Good and Bad states:
  - Good state: no packet loss
  - Bad state:  high probability of packet loss
Transitions: p_gb (Good→Bad), p_bg (Bad→Good)

Essential components (rank 0..min_rank-1) are always preserved.
Components rank min_rank..R-1 are subject to GE-modeled packet loss.

Usage:
    python simulate_packetloss.py \
        -frame_path "data/sky" \
        -prompt_dir "data/sky/results/rank16_interval10" \
        -rank 16 \
        -interval 10 \
        --min_rank 4 \
        --loss_rates 0.0 0.1 0.2 0.3 0.4 0.5
"""

import os
import re
import argparse
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from glob import glob
from diffusers import AutoencoderTiny
from torchvision.utils import save_image
from scripts.demo.streamlit_helpers import *
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from quantization import QParam
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

VERSION2SPECS = {
    "SD-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/sd_turbo.safetensors",
    },
}


class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
            ]
        uc = cond
        x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc


def seeded_randn(shape, seed):
    randn = np.random.RandomState(seed).randn(*shape)
    randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
    return randn


class SeededNoise:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x):
        self.seed = self.seed + 1
        return seeded_randn(x.shape, self.seed)


def slerp(a, b, t, eps=1e-5):
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    cos_theta = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    use_lerp = sin_theta < eps
    factor_a = torch.sin((1.0 - t) * theta) / sin_theta
    factor_b = torch.sin(t * theta) / sin_theta
    result = factor_a * a + factor_b * b
    lerp_result = (1.0 - t) * a + t * b
    result = torch.where(use_lerp.expand_as(result), lerp_result, result)
    return result


def load_image(path):
    """Load an image and convert to numpy array (H, W, C) in [0, 1] range."""
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def gilbert_elliott_loss_mask(n_components, loss_rate, p_gb=0.1, p_bg=0.3, seed=42):
    """Generate a binary loss mask using the Gilbert-Elliott burst-loss model.

    Two-state Markov chain:
      - Good (0): no packet loss.
      - Bad  (1): packets drop with probability loss_bad.

    loss_bad is derived from the target average loss_rate:
        loss_rate = P(bad) × loss_bad
        P(bad)    = p_gb / (p_gb + p_bg)

    Args:
        n_components: number of components (packets) to generate mask for.
        loss_rate:    target average packet loss rate in [0, 1].
        p_gb:         Good → Bad transition probability.
        p_bg:         Bad → Good transition probability.
        seed:         random seed for reproducibility.

    Returns:
        mask: boolean array of shape (n_components,), True = lost.
    """
    rng = np.random.RandomState(seed)

    # Steady-state probability of Bad state
    p_bad = p_gb / (p_gb + p_bg)

    # Derive loss probability in Bad state to hit the target average loss rate.
    # Assume loss_good = 0.
    if p_bad > 0:
        loss_bad = min(loss_rate / p_bad, 1.0)
    else:
        loss_bad = 0.0

    state = 0  # start in Good state
    mask = np.zeros(n_components, dtype=bool)

    for i in range(n_components):
        if state == 0:  # Good
            mask[i] = False
            if rng.rand() < p_gb:
                state = 1
        else:  # Bad
            mask[i] = rng.rand() < loss_bad
            if rng.rand() < p_bg:
                state = 0

    return mask


def apply_packet_loss(U, V, min_rank, loss_rate, p_gb=0.1, p_bg=0.3, seed=42):
    """Apply GE packet loss to SVD components beyond min_rank.

    Components 0 .. min_rank-1 are always preserved (essential low-rank).
    Components min_rank .. R-1 are subject to GE burst loss — lost columns
    of U and rows of V are set to zero.

    Args:
        U, V:      dequantized prompt matrices.
        min_rank:  number of essential components to always keep.
        loss_rate: target average packet loss rate.
        p_gb, p_bg: GE transition probabilities.
        seed:      random seed.

    Returns:
        U_lost, V_lost: tensors with lost components zeroed out.
    """
    rank = U.shape[1]
    if min_rank >= rank:
        return U, V

    n_extra = rank - min_rank
    mask_extra = gilbert_elliott_loss_mask(n_extra, loss_rate, p_gb, p_bg, seed)

    # Full keep mask: first min_rank always True, rest from GE model
    full_keep = np.ones(rank, dtype=bool)
    full_keep[min_rank:] = ~mask_extra  # True = kept, False = lost

    U_lost = U.clone()
    V_lost = V.clone()

    lost_indices = np.where(~full_keep)[0]
    if len(lost_indices) > 0:
        U_lost[:, lost_indices] = 0
        V_lost[lost_indices, :] = 0

    return U_lost, V_lost


@torch.no_grad()
def generate_video_with_loss(
        model, sampler, decoder, prompt_dir, frame_path,
        interval, min_rank, train_rank, loss_rate,
        p_gb=0.1, p_bg=0.3, slerp_mode=True
):
    """Generate video frames with simulated packet loss on each keyframe pair.

    Returns:
        frames: dict mapping frame_id → generated image (H, W, 3) float32 [0, 1].
    """
    prompt_dir_full = prompt_dir

    H, W = 512, 512
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    uc = None
    rand_noise = seeded_randn(shape, 88)
    sigma = torch.Tensor([0.05]).float().cuda()

    def denoiser(input, sigma, c):
        return model.denoiser(model.model, input, sigma, c)

    prev_frame = None
    frames = {}

    loss_seed = 42

    prompts = sorted(glob(os.path.join(prompt_dir_full, 'frame_*.prompt')))
    for prompt_pair in zip(prompts[::], prompts[1::]):
        prompt_curr = prompt_pair[0]
        id_curr = int(re.search(r'frame_(\d{5})\.prompt', prompt_curr).group(1))
        prompt_next = prompt_pair[1]
        id_next = int(re.search(r'frame_(\d{5})\.prompt', prompt_next).group(1))

        prompt_curr_data = torch.load(prompt_curr, weights_only=True)
        prompt_next_data = torch.load(prompt_next, weights_only=True)

        U_curr, V_curr = prompt_curr_data['U'], prompt_curr_data['V']
        U_next, V_next = prompt_next_data['U'], prompt_next_data['V']

        # Dequantize
        def dequantize(U_q, V_q, prompt_data):
            qp_u = QParam(num_bits=8)
            qp_u.scale = prompt_data['U_scale']
            qp_u.zero_point = prompt_data['U_zero_point']
            U = qp_u.dequantize_tensor(U_q)
            qp_v = QParam(num_bits=8)
            qp_v.scale = prompt_data['V_scale']
            qp_v.zero_point = prompt_data['V_zero_point']
            V = qp_v.dequantize_tensor(V_q)
            return U, V

        U_curr, V_curr = dequantize(U_curr, V_curr, prompt_curr_data)
        U_next, V_next = dequantize(U_next, V_next, prompt_next_data)

        # ---- Apply packet loss to components beyond min_rank ----
        U_curr, V_curr = apply_packet_loss(
            U_curr, V_curr, min_rank, loss_rate, p_gb, p_bg, seed=loss_seed
        )
        loss_seed += 1
        U_next, V_next = apply_packet_loss(
            U_next, V_next, min_rank, loss_rate, p_gb, p_bg, seed=loss_seed
        )
        loss_seed += 1

        eff_rank = train_rank  # normalise by full training rank

        if prev_frame is None:
            prev_frame = torch.load(os.path.join(prompt_dir_full, 'init.pth'), weights_only=True)
            z = (prev_frame * sigma + rand_noise * (1 - sigma))
            c = (U_curr @ V_curr / np.sqrt(eff_rank)).unsqueeze(dim=0)
            prompt = {'crossattn': c}
            samples_z = sampler(denoiser, z, cond=prompt, uc=uc)
            img = decoder(samples_z)
            img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
            frames[id_curr] = img[0].permute(1, 2, 0).cpu().numpy()
            prev_frame = samples_z

        z = (prev_frame * sigma + rand_noise * (1 - sigma))
        for step in range(1, interval + 1):
            t = step / interval
            t_i = torch.tensor(t, device=U_curr.device)

            if slerp_mode:
                u = slerp(U_curr.T, U_next.T, t_i.view(-1, 1)).T
                v = slerp(V_curr, V_next, t_i.view(-1, 1))
            else:
                u = (1 - t_i.view(1, -1)) * U_curr + t_i.view(1, -1) * U_next
                v = (1 - t_i.view(-1, 1)) * V_curr + t_i.view(-1, 1) * V_next

            c = (u @ v / np.sqrt(eff_rank)).unsqueeze(dim=0)
            prompt = {'crossattn': c}
            samples_z = sampler(denoiser, z, cond=prompt, uc=uc)
            img = decoder(samples_z)
            img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
            frame_id = id_curr + step
            frames[frame_id] = img[0].permute(1, 2, 0).cpu().numpy()
            prev_frame = samples_z
            z = (prev_frame * sigma + rand_noise * (1 - sigma))

    return frames


def evaluate_frames(frames, frame_path, gt_total_ids):
    """Compute PSNR / SSIM / LPIPS for generated frames vs ground truth."""
    psnr_list, ssim_list, lpips_list = [], [], []

    loss_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

    for f_id, gen_img in sorted(frames.items()):
        if f_id >= gt_total_ids:
            continue
        gt_path = os.path.join(frame_path, '{:05d}.png'.format(f_id))
        if not os.path.exists(gt_path):
            continue
        gt_img = load_image(gt_path)

        p = psnr(gen_img, gt_img, data_range=1.0)
        s = ssim(gen_img, gt_img, data_range=1.0, channel_axis=-1)

        t1 = torch.from_numpy(gen_img).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
        t2 = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
        l = loss_lpips(t1, t2).item()

        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)

    return {
        'psnr_mean': np.mean(psnr_list),
        'psnr_std':  np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std':  np.std(ssim_list),
        'lpips_mean': np.mean(lpips_list),
        'lpips_std':  np.std(lpips_list),
        'n_frames':   len(psnr_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-frame_path', type=str, default="data/sky",
                        help='Path to video frames directory')
    parser.add_argument('-prompt_dir', type=str, default=None,
                        help='Path to prompt directory (overrides auto-resolve)')
    parser.add_argument('-rank', type=int, default=16,
                        help='Training rank')
    parser.add_argument('-interval', type=int, default=10,
                        help='Keyframe interval')
    parser.add_argument('-max_id', type=int, default=140,
                        help='Maximum frame ID')
    parser.add_argument('--min_rank', type=int, default=4,
                        help='Essential rank always preserved (no packet loss)')
    parser.add_argument('--loss_rates', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help='List of target packet loss rates to evaluate')
    parser.add_argument('--p_gb', type=float, default=0.1,
                        help='GE model: Good → Bad transition probability')
    parser.add_argument('--p_bg', type=float, default=0.3,
                        help='GE model: Bad → Good transition probability')
    parser.add_argument('--slerp', action='store_true', default=True)
    parser.add_argument('--no-slerp', action='store_false', dest='slerp')
    args = parser.parse_args()

    # Resolve prompt directory
    if args.prompt_dir is None:
        prompt_dir = os.path.join(
            args.frame_path,
            'results/rank{}_interval{}'.format(args.rank, args.interval)
        )
    else:
        prompt_dir = args.prompt_dir

    print("=" * 70)
    print("Packet Loss Simulation: Quality vs Loss Rate (Gilbert-Elliott Model)")
    print("=" * 70)
    print(f"  Frame path:    {args.frame_path}")
    print(f"  Prompt dir:    {prompt_dir}")
    print(f"  Training rank: {args.rank}")
    print(f"  Min rank:      {args.min_rank}")
    print(f"  Interval:      {args.interval}")
    print(f"  Max ID:        {args.max_id}")
    print(f"  Loss rates:    {args.loss_rates}")
    print(f"  GE params:     p_gb={args.p_gb}, p_bg={args.p_bg}")
    print("=" * 70)

    # Load models once
    version_dict = VERSION2SPECS['SD-Turbo']
    state = init_st(version_dict, load_filter=True)
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]
    load_model(model)
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).cuda()
    sampler = SubstepSampler(
        n_sample_steps=1,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    seed_ = 88
    sampler.noise_sampler = SeededNoise(seed=seed_)

    results = {}
    for loss_rate in sorted(args.loss_rates):
        print(f"\n--- Generating with loss_rate={loss_rate:.2f} ---")
        frames = generate_video_with_loss(
            model, sampler, decoder=taesd.decoder,
            prompt_dir=prompt_dir, frame_path=args.frame_path,
            interval=args.interval, min_rank=args.min_rank,
            train_rank=args.rank, loss_rate=loss_rate,
            p_gb=args.p_gb, p_bg=args.p_bg,
            slerp_mode=args.slerp,
        )
        metrics = evaluate_frames(frames, args.frame_path, args.max_id)
        results[loss_rate] = metrics
        print(f"  loss_rate={loss_rate:.2f}: "
              f"PSNR={metrics['psnr_mean']:.4f}±{metrics['psnr_std']:.4f}, "
              f"SSIM={metrics['ssim_mean']:.4f}±{metrics['ssim_std']:.4f}, "
              f"LPIPS={metrics['lpips_mean']:.4f}±{metrics['lpips_std']:.4f}, "
              f"frames={metrics['n_frames']}")

        # Save first frame to image_demo
        save_dir = "/root/autodl-tmp/image_demo"
        os.makedirs(save_dir, exist_ok=True)
        first_fid = min(frames.keys())
        first_img_tensor = torch.from_numpy(frames[first_fid]).permute(2, 0, 1).unsqueeze(0)
        save_path = os.path.join(save_dir, f"frame_loss{loss_rate:.2f}.png")
        save_image(first_img_tensor, save_path)
        print(f"  Saved first frame (id={first_fid}) to {save_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Quality vs Packet Loss Rate")
    print("=" * 70)
    print(f"{'Loss':>8} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8} | {'Frames':>6}")
    print("-" * 70)
    for loss_rate in sorted(results.keys()):
        m = results[loss_rate]
        print(f"{loss_rate:8.2f} | "
              f"{m['psnr_mean']:8.4f} | "
              f"{m['ssim_mean']:8.4f} | "
              f"{m['lpips_mean']:8.4f} | "
              f"{m['n_frames']:6d}")

    # Estimated overhead comparison
    print("\n" + "=" * 70)
    print("Estimated effective components (min_rank + surviving extras)")
    print("=" * 70)
    print(f"{'Loss':>8} | {'Eff_rank':>10} | {'Surviving':>10}")
    print("-" * 50)
    for loss_rate in sorted(results.keys()):
        if loss_rate == 0.0:
            eff = args.rank
        else:
            # Expected surviving: min_rank + (rank-min_rank)*(1-loss_rate)
            eff = args.min_rank + (args.rank - args.min_rank) * (1 - loss_rate)
        print(f"{loss_rate:8.2f} | {eff:10.2f} | {eff / args.rank * 100:9.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
