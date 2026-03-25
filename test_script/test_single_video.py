import argparse
import os

import cv2
import numpy as np
import natsort
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from diffsynth import save_video
from diffsynth.util.alignment import disparity2depth
from examples.wanvideo.model_training.WanTrainingModule import \
    WanTrainingModule


# =============================
# Helper: Math & Alignment
# =============================
def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
    """Computes scale and shift for overlap alignment."""
    if mask is None:
        mask = np.ones_like(ref_frames)

    a_00 = np.sum(mask * curr_frames * curr_frames)
    a_01 = np.sum(mask * curr_frames)
    a_11 = np.sum(mask)
    b_0 = np.sum(mask * curr_frames * ref_frames)
    b_1 = np.sum(mask * ref_frames)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale, shift = 1.0, 0.0

    return scale, shift


# =============================
# Helper: Video Processing
# =============================
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(
        video_np).permute(0, 3, 1, 2).float() / 255.0

    # No per-file names; COLMAP alignment uses sorted image order (see match_colmap_images).
    return video_tensor.unsqueeze(0), fps, None


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _list_image_paths(folder_path):
    """Sorted list of image file paths under folder_path (non-recursive)."""
    paths = []
    for name in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(name)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            paths.append(os.path.join(folder_path, name))
    return paths


def read_image_sequence(folder_path, fps):
    """Loads RGB frames from a directory; same tensor layout as read_video.

    Returns:
        video_tensor: [1, T, C, H, W]
        fps: float
        frame_keys: list[str] of basenames (for COLMAP name matching)
    """
    image_paths = _list_image_paths(folder_path)
    if not image_paths:
        raise ValueError(
            f"No supported images in {folder_path} "
            f"(extensions: {sorted(_IMAGE_EXTENSIONS)})")

    frame_keys = [os.path.basename(p) for p in image_paths]
    frames = []
    for p in image_paths:
        frame = cv2.imread(p, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot read image: {p}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(
        video_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor.unsqueeze(0), float(fps), frame_keys


def resize_for_training_scale(video_tensor, target_h=480, target_w=640):
    B, T, C, H, W = video_tensor.shape
    ratio = max(target_h / H, target_w / W)
    new_H = int(np.ceil(H * ratio))
    new_W = int(np.ceil(W * ratio))

    # Align to 16
    new_H = (new_H + 15) // 16 * 16
    new_W = (new_W + 15) // 16 * 16

    if new_H == H and new_W == W:
        return video_tensor, (H, W)

    video_reshape = video_tensor.view(B * T, C, H, W)
    resized = F.interpolate(video_reshape, size=(
        new_H, new_W), mode="bilinear", align_corners=False)
    resized = resized.view(B, T, C, new_H, new_W)
    return resized, (H, W)


def resize_depth_back(depth_np, orig_size):
    orig_H, orig_W = orig_size
    depth_tensor = torch.from_numpy(depth_np).permute(0, 3, 1, 2).float()
    depth_tensor = F.interpolate(depth_tensor, size=(
        orig_H, orig_W), mode='bilinear', align_corners=False)
    return depth_tensor.permute(0, 2, 3, 1).cpu().numpy()


# =============================
# COLMAP: calibration & point clouds
# =============================
def _resolve_colmap_sparse_path(colmap_root):
    """Return directory containing cameras.bin/txt (sparse model root)."""
    root = os.path.abspath(colmap_root)
    candidates = [
        root,
        os.path.join(root, "sparse", "0"),
        os.path.join(root, "0"),
    ]
    for c in candidates:
        if os.path.isdir(c) and (
            os.path.isfile(os.path.join(c, "cameras.bin"))
            or os.path.isfile(os.path.join(c, "cameras.txt"))
        ):
            return c
    raise FileNotFoundError(
        f"No COLMAP sparse model (cameras.bin/txt) under {colmap_root!r}; "
        "tried the path itself and sparse/0."
    )


def _colmap_image_lookup(reconstruction):
    """basename(lower) -> first Image (COLMAP names may include subdirs)."""
    by_base = {}
    by_lower = {}
    for im in reconstruction.images.values():
        b = os.path.basename(im.name)
        by_base.setdefault(b, im)
        by_lower.setdefault(b.lower(), im)
    return by_base, by_lower


def match_colmap_images(reconstruction, num_frames, frame_keys):
    """List of (frame_index, pycolmap.Image) for frames present in COLMAP.

    If frame_keys is a list of basenames, match by name (exact, then case-fold).
    If frame_keys is None (video input), match by natsorted COLMAP image order
    aligned to frame index (same length required).
    """
    images = list(reconstruction.images.values())
    if not images:
        raise ValueError("COLMAP reconstruction has no registered images.")

    if frame_keys is not None:
        by_base, by_lower = _colmap_image_lookup(reconstruction)
        pairs = []
        for t in range(num_frames):
            k = frame_keys[t]
            im = by_base.get(k) or by_lower.get(k.lower())
            if im is not None:
                pairs.append((t, im))
        return pairs

    sorted_imgs = natsort.natsorted(images, key=lambda im: im.name)
    if len(sorted_imgs) != num_frames:
        print(
            f"Warning: COLMAP has {len(sorted_imgs)} images but input has "
            f"{num_frames} frames; aligning first {min(len(sorted_imgs), num_frames)} "
            "by sorted image name (video mode)."
        )
    n = min(len(sorted_imgs), num_frames)
    return [(t, sorted_imgs[t]) for t in range(n)]


def _camera_for_image(reconstruction, image):
    import pycolmap

    cam = reconstruction.cameras[image.camera_id]
    return pycolmap.Camera(**cam.todict())


def _depth_hw_z(depth_frame):
    """(H, W) or (H, W, C) -> single-channel map (H, W), float64."""
    if depth_frame.ndim == 2:
        return depth_frame.astype(np.float64, copy=False)
    if depth_frame.ndim == 3:
        return np.mean(depth_frame, axis=-1).astype(np.float64, copy=False)
    raise ValueError(f"Unexpected depth shape {depth_frame.shape}")


def _to_perpendicular_z(depth_hw, depth_space):
    """Map network output to perpendicular camera-frame Z for pinhole unprojection.

    Training/validation treats the decoder output as *disparity* (same space as
    ``depth2disparity(gt_depth)`` = 1/Z), then uses ``disparity2depth`` for metrics.
    Using that signal directly as Z bends the cloud and breaks multi-view alignment.
    """
    if depth_space == "z":
        return depth_hw
    if depth_space == "disparity":
        disp = np.maximum(depth_hw, 1e-6)
        return disparity2depth(disp)
    raise ValueError(f"Unknown depth_space: {depth_space!r}")


def depth_to_world_points(depth_hw, camera, world_from_cam, stride=1):
    """Unproject depth to 3D points in world frame (COLMAP convention).

    depth_hw: metric/relative Z in camera coordinates (along optical axis),
    same resolution as camera after rescale.
    """
    H, W = depth_hw.shape
    if camera.width != W or camera.height != H:
        raise ValueError(
            f"Camera size {camera.width}x{camera.height} != depth {W}x{H}"
        )

    xs = np.arange(0, W, stride, dtype=np.float64)
    ys = np.arange(0, H, stride, dtype=np.float64)
    uu, vv = np.meshgrid(xs, ys)
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
    z = depth_hw[vv.astype(int), uu.astype(int)].ravel()

    xy = camera.cam_from_img(uv)
    x_cam = xy[:, 0] * z
    y_cam = xy[:, 1] * z
    z_cam = z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    valid = np.isfinite(z) & (z > 0)
    pts_cam = pts_cam[valid]
    if pts_cam.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pts_world = world_from_cam * pts_cam
    return np.asarray(pts_world, dtype=np.float64)


def _write_ply_ascii(path, points_xyz):
    """Minimal ASCII PLY writer (x y z per vertex)."""
    n = points_xyz.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points_xyz[i]
            f.write(f"{x} {y} {z}\n")


def _safe_stem_from_image_name(name):
    base = os.path.splitext(os.path.basename(name))[0]
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in base)
    return out or "frame"


def save_colmap_depth_point_clouds(
    depth,
    reconstruction,
    frame_matches,
    output_dir,
    pts_stride=1,
    depth_space="disparity",
):
    """Save one PLY per matched frame under output_dir/pts/."""
    pts_dir = os.path.join(output_dir, "pts")
    os.makedirs(pts_dir, exist_ok=True)

    saved = []
    for t, image in frame_matches:
        cam = _camera_for_image(reconstruction, image)
        depth_hw = _depth_hw_z(depth[t])
        depth_hw = _to_perpendicular_z(depth_hw, depth_space)
        orig_h, orig_w = depth_hw.shape
        cam.rescale(orig_w, orig_h)

        world_from_cam = image.cam_from_world().inverse()
        pts = depth_to_world_points(depth_hw, cam, world_from_cam, stride=pts_stride)
        stem = _safe_stem_from_image_name(image.name)
        out_path = os.path.join(pts_dir, f"{t:06d}_{stem}.ply")
        _write_ply_ascii(out_path, pts)
        saved.append(out_path)
        print(f"Wrote {out_path} ({pts.shape[0]} points)")

    return saved


def run_colmap_point_export(args, depth, frame_keys):
    if not args.colmap:
        return
    try:
        import pycolmap
    except ImportError as e:
        raise ImportError(
            "pycolmap is required for --colmap. Install with: pip install pycolmap"
        ) from e

    sparse_path = _resolve_colmap_sparse_path(args.colmap)
    reconstruction = pycolmap.Reconstruction(sparse_path)
    T = depth.shape[0]
    matches = match_colmap_images(reconstruction, T, frame_keys)
    if not matches:
        print("No input frames matched COLMAP images; skipping pts/ export.")
        return

    print(
        f"COLMAP: {len(matches)} / {T} frames matched "
        f"(model from {sparse_path})."
    )
    print(
        f"COLMAP unprojection: treating network output as {args.colmap_depth_space!r} "
        "(use --colmap_depth_space z if your checkpoint decodes metric Z)."
    )
    save_colmap_depth_point_clouds(
        depth,
        reconstruction,
        matches,
        args.output_dir,
        pts_stride=args.pts_stride,
        depth_space=args.colmap_depth_space,
    )


def pad_time_mod4(video_tensor):
    """Pads the temporal dimension to satisfy 4n+1 requirement."""
    B, T, C, H, W = video_tensor.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = video_tensor[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, pad_frames], dim=1)
    return video_tensor, T


def get_window_index(T, window_size, overlap):
    if T <= window_size:
        return [(0, T)]
    res = [(0, window_size)]
    start = window_size - overlap
    while start < T:
        end = start + window_size
        if end < T:
            res.append((start, end))
            start += window_size - overlap
        else:
            # Last window ensures full window_size length if possible
            start = max(0, T - window_size)
            res.append((start, T))
            break
    return res


# =============================
# Core Inference
# =============================
def generate_depth_sliced(model, input_rgb, window_size=45, overlap=9, scale_only=False):
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    print(f"depth_windows {depth_windows}")

    depth_res_list = []

    # 1. Inference per window
    for start, end in tqdm(depth_windows, desc="Inferencing Slices"):
        _input_rgb_slice = input_rgb[:, start:end]

        # Ensure 4n+1 padding
        _input_rgb_slice, origin_T = pad_time_mod4(_input_rgb_slice)
        _input_frame = _input_rgb_slice.shape[1]
        _input_height, _input_width = _input_rgb_slice.shape[-2:]

        outputs = model.pipe(
            prompt=[""] * B,
            negative_prompt=[""] * B,
            mode=model.args.mode,
            height=_input_height,
            width=_input_width,
            num_frames=_input_frame,
            batch_size=B,
            input_image=_input_rgb_slice[:, 0],
            extra_images=_input_rgb_slice,
            extra_image_frame_index=torch.ones(
                [B, _input_frame]).to(model.pipe.device),
            input_video=_input_rgb_slice,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        # Drop the padded frames
        depth_res_list.append(outputs['depth'][:, :origin_T])

    # 2. Overlap Alignment
    depth_list_aligned = None
    prev_end = None

    for i, (t, (start, end)) in enumerate(zip(depth_res_list, depth_windows)):
        print(f"Handling window {i} start: {start}, end: {end}")

        if i == 0:
            depth_list_aligned = t
            prev_end = end
            continue

        curr_start = start
        real_overlap = prev_end - curr_start

        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]
            curr_frames = t[:, :real_overlap]

            if scale_only:
                scale = np.sum(curr_frames * ref_frames) / \
                    (np.sum(curr_frames * curr_frames) + 1e-6)
                shift = 0.0
            else:
                scale, shift = compute_scale_and_shift(curr_frames, ref_frames)

            scale = np.clip(scale, 0.7, 1.5)

            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0

            # Debugging Output
            curr_overlap_aligned = aligned_t[:, :real_overlap]
            diff = np.abs(curr_overlap_aligned - ref_frames)
            mae_scalar = float(
                diff.mean(axis=tuple(range(1, diff.ndim))).mean())

            print(f"\n[Overlap {i}]")
            print(f"real_overlap = {real_overlap}")
            print(f"scale = {scale:.8f}, shift = {shift:.8f}")
            print(
                f"aligned curr range = {aligned_t.min():.6f} ~ {aligned_t.max():.6f}")
            print(f"overlap MAE(after align) = {mae_scalar:.6f}")

            # Smooth blending
            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(
                1, real_overlap, 1, 1, 1)
            smooth_overlap = (1 - alpha) * ref_frames + \
                alpha * aligned_t[:, :real_overlap]

            depth_list_aligned = np.concatenate(
                [depth_list_aligned[:, :-real_overlap], smooth_overlap,
                 aligned_t[:, real_overlap:]], axis=1
            )
        else:
            # Fallback if no overlap exists
            depth_list_aligned = np.concatenate(
                [depth_list_aligned, t], axis=1)

        print(
            f"Total depth range after concat = {depth_list_aligned.min():.6f} ~ {depth_list_aligned.max():.6f}")
        prev_end = end

    # Crop to original length
    return depth_list_aligned[:, :T]


# =============================
# Pipeline Components
# =============================
def load_model(ckpt_dir, yaml_args):
    """Initializes and loads the model checkpoint."""
    accelerator = Accelerator()
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=yaml_args.lora_rank,
        lora_base_model=yaml_args.lora_base_model,
        args=yaml_args,
    )

    ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
    state_dict = load_file(ckpt_path, device="cpu")
    dit_state_dict = {k.replace("pipe.dit.", ""): v for k,
                      v in state_dict.items() if "pipe.dit." in k}
    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    model.merge_lora_layer()
    model = model.to("cuda")
    
    return model


def load_video_data(args):
    """Loads and resizes the input video or an image sequence from a folder."""
    if os.path.isdir(args.input_video):
        input_tensor, origin_fps, frame_keys = read_image_sequence(
            args.input_video, args.sequence_fps)
        print(f"Loaded {input_tensor.shape[1]} frames from image sequence")
    else:
        input_tensor, origin_fps, frame_keys = read_video(args.input_video)
    print("Original shape:", input_tensor.shape)

    input_tensor, orig_size = resize_for_training_scale(
        input_tensor, args.height, args.width)
    print("Resized shape:", input_tensor.shape)
    print(f"input range {input_tensor.min()} - {input_tensor.max()}")

    return input_tensor, orig_size, origin_fps, frame_keys


def predict_depth(model, input_tensor, orig_size, args):
    """Runs depth prediction and post-processes the output to original size."""
    depth = generate_depth_sliced(
        model, input_tensor, args.window_size, args.overlap)[0]
    print(f"depth range shape {depth.min()} - {depth.max()}, shape {depth.shape}")

    # Post Process: resize back to original
    depth = resize_depth_back(depth, orig_size)
    print(f"after resizing {depth.min()} - {depth.max()}, {depth.shape}")

    return depth


def save_results(depth, origin_fps, args):
    """Normalizes and saves the depth video to disk."""
    os.makedirs(args.output_dir, exist_ok=True)
    in_path = args.input_video
    if os.path.isdir(in_path):
        base_name = os.path.basename(os.path.normpath(in_path))
    else:
        base_name = os.path.splitext(os.path.basename(in_path))[0]
    gray_scale = 'gray' if args.grayscale else 'color'
    out_prefix = os.path.join(
        args.output_dir, f"{base_name}_{gray_scale}")

    output_path = f"{out_prefix}_depth_vis.mp4"
    print(f"Saving to {output_path}")
    d_min, d_max = depth.min(), depth.max()
    vis_depth = (depth - d_min) / (d_max - d_min + 1e-8)
    
    save_video(vis_depth, output_path,
               fps=origin_fps, quality=6, grayscale=args.grayscale)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to a video file, or a directory of images (sorted by filename).",
    )
    parser.add_argument(
        "--sequence_fps",
        type=float,
        default=24.0,
        help="Output video FPS when --input_video is a folder of images.",
    )
    parser.add_argument("--output_dir", type=str,
                        default="./inference_results")
    parser.add_argument('--model_config', default='ckpt/model_config.yaml')
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument("--overlap", type=int, default=9)
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument(
        "--colmap",
        type=str,
        default=None,
        help="COLMAP reconstruction root (expects sparse model at this path or sparse/0). "
        "Exports one PLY point cloud per matched frame under output_dir/pts/.",
    )
    parser.add_argument(
        "--pts_stride",
        type=int,
        default=1,
        help="Pixel stride when sampling depth for COLMAP point clouds (larger = fewer points).",
    )
    parser.add_argument(
        "--colmap_depth_space",
        type=str,
        choices=("disparity", "z"),
        default="disparity",
        help="Network output semantics before unprojection: 'disparity' (1/Z space, DVD/Wan "
        "validation default) or 'z' (already perpendicular depth in camera frame).",
    )
    return parser.parse_args()


# =============================
# Main Script
# =============================
def main():
    args = parse_args()
    yaml_args = OmegaConf.load(args.model_config)

    # 1. Load Model
    model = load_model(args.ckpt, yaml_args)

    # 2. Load Video
    input_tensor, orig_size, origin_fps, frame_keys = load_video_data(args)

    # 3. Predict Depth
    depth = predict_depth(model, input_tensor, orig_size, args)

    # 4. Save Results
    save_results(depth, origin_fps, args)

    # 5. Optional COLMAP-registered world point clouds
    run_colmap_point_export(args, depth, frame_keys)

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()