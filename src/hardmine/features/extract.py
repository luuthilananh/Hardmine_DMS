"""Trích embedding một lần cho các cửa sổ video (bước 2). Cần extras: `pip install -e ".[gpu,video]"`.

Hỗ trợ hai loại backbone Hugging Face:
  - kind="video": VideoMAE / V-JEPA 2 (AutoModel + AutoImageProcessor/AutoVideoProcessor), mean-pool token cuối.
  - kind="image": SigLIP / SigLIP 2 / CLIP, `get_image_features` trên `frames_per_window` frame, lấy trung bình.

Đầu ra (resume được):
  <out_dir>/emb_<shard>.npy   float16, (n, d)
  <out_dir>/index.csv         window_id, shard, row
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

PRESETS = {
    "videomae_base": ("MCG-NJU/videomae-base", "video", 16),
    "vjepa2_vitl": ("facebook/vjepa2-vitl-fpc64-256", "video", 16),
    "siglip2_base": ("google/siglip2-base-patch16-224", "image", 4),
    "tiny_test": ("hf-internal-testing/tiny-random-VideoMAEModel", "video", 16),
}


def read_frames(video_path: str, t_start: float, t_end: float, n: int) -> list[np.ndarray]:
    """Đọc n frame lấy đều trong [t_start, t_end) (giây). Ảnh 1 kênh (NIR) được lặp thành 3 kênh RGB."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(t_start * fps, max(t_start * fps, t_end * fps - 1), n).round().astype(int)
    idx = np.clip(idx, 0, max(total - 1, 0))
    frames, last = [], None
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            if last is None:
                raise RuntimeError(f"Không đọc được frame {i} của {video_path}")
            f = last
        last = f
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(f)
    cap.release()
    return frames


@dataclass
class Embedder:
    preset: str = "videomae_base"
    device: str | None = None
    fp16: bool = True

    def __post_init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.model_id, self.kind, self.n_frames = PRESETS[self.preset]
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if (self.fp16 and self.device == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device).eval()
        try:
            self.proc = AutoImageProcessor.from_pretrained(self.model_id)
        except Exception:  # một số model video dùng AutoVideoProcessor
            from transformers import AutoVideoProcessor
            self.proc = AutoVideoProcessor.from_pretrained(self.model_id)
        self.dtype = dtype
        cfg = getattr(self.model.config, "num_frames", None)
        if self.kind == "video" and cfg:
            self.n_frames = int(cfg)

    def embed(self, batch_frames: list[list[np.ndarray]]) -> np.ndarray:
        import torch
        with torch.inference_mode():
            if self.kind == "video":
                inputs = self.proc(batch_frames, return_tensors="pt")
                key = "pixel_values" if "pixel_values" in inputs else next(iter(inputs))
                x = inputs[key].to(self.device, self.dtype)
                out = self.model(**{key: x})
                h = out.last_hidden_state.float().mean(1)
            else:
                flat = [f for frames in batch_frames for f in frames]
                inputs = self.proc(images=flat, return_tensors="pt")
                x = inputs["pixel_values"].to(self.device, self.dtype)
                feats = self.model.get_image_features(pixel_values=x).float()
                h = feats.view(len(batch_frames), -1, feats.shape[-1]).mean(1)
        return h.cpu().numpy()


def extract(windows: pd.DataFrame, video_path_of, out_dir: str, preset: str = "videomae_base",
            batch_size: int = 8, shard_size: int = 2048, device: str | None = None) -> pathlib.Path:
    """windows cần cột: window_id, session_id, t_start, t_end. `video_path_of(row)` trả đường dẫn video.

    Resume: bỏ qua window_id đã có trong index.csv.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "index.csv"
    done = set(pd.read_csv(idx_path)["window_id"]) if idx_path.exists() else set()
    todo = windows[~windows["window_id"].isin(done)]
    if todo.empty:
        return out
    emb = Embedder(preset=preset, device=device)
    shard = len(list(out.glob("emb_*.npy")))
    buf, ids = [], []

    def flush():
        nonlocal shard, buf, ids
        if not buf:
            return
        arr = np.concatenate(buf).astype(np.float16)
        np.save(out / f"emb_{shard:05d}.npy", arr)
        rec = pd.DataFrame({"window_id": ids, "shard": shard, "row": np.arange(len(ids))})
        rec.to_csv(idx_path, mode="a", header=not idx_path.exists(), index=False)
        shard += 1
        buf, ids = [], []

    rows = list(todo.itertuples(index=False))
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        frames = [read_frames(video_path_of(r), r.t_start, r.t_end, emb.n_frames) for r in chunk]
        buf.append(emb.embed(frames))
        ids += [r.window_id for r in chunk]
        if sum(len(b) for b in buf) >= shard_size:
            flush()
    flush()
    return out


def load(out_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Trả (window_ids, emb float32) theo thứ tự index.csv."""
    out = pathlib.Path(out_dir)
    idx = pd.read_csv(out / "index.csv")
    shards = {s: np.load(out / f"emb_{s:05d}.npy") for s in idx["shard"].unique()}
    emb = np.stack([shards[s][r] for s, r in zip(idx["shard"], idx["row"])]).astype(np.float32)
    return idx["window_id"].to_numpy(), emb


def windows_from_folder(video_dir: str, length_s: float = 2.0, stride_s: float = 1.0,
                        pattern: str = "**/*.mp4") -> pd.DataFrame:
    """Tạo bảng cửa sổ cho một thư mục video bất kỳ (dữ liệu tự quay, Kaggle). driver_id = tên thư mục cha."""
    import cv2
    rows, wid = [], 0
    for p in sorted(pathlib.Path(video_dir).glob(pattern)):
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
        cap.release()
        t = 0.0
        while t + length_s <= dur + 1e-6:
            rows.append({"window_id": wid, "session_id": p.stem, "driver_id": p.parent.name,
                         "video_path": str(p), "t_start": t, "t_end": t + length_s})
            wid += 1
            t += stride_s
    return pd.DataFrame(rows)
