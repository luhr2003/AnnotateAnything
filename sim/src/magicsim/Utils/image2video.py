import os
import re
import argparse
from typing import List, Tuple, Optional

import cv2


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(s: str):
    # 自然排序：img2 < img10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder: str) -> List[str]:
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(folder, name))
    files.sort(key=lambda p: natural_key(os.path.basename(p)))
    return files


def read_image(path: str) -> Optional[cv2.Mat]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def parse_size(size_str: str) -> Tuple[int, int]:
    # e.g. "1280x720"
    if "x" not in size_str:
        raise ValueError("size must be like 1280x720")
    w, h = size_str.lower().split("x")
    return int(w), int(h)


def resize_to(img, w: int, h: int, keep_aspect: bool):
    if not keep_aspect:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # 保持比例：letterbox
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = cv2.cvtColor(
        cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
    )  # dummy to get type; will overwrite below
    canvas = 0 * canvas  # black background
    canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)

    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Convert an image folder to a video.")
    ap.add_argument(
        "--input_dir", required=True, help="Path to folder containing images."
    )
    ap.add_argument("--output", required=True, help="Output video path, e.g. out.mp4")
    ap.add_argument("--fps", type=float, default=30.0, help="Frames per second.")
    ap.add_argument(
        "--codec", default="mp4v", help="FourCC codec: mp4v, avc1, XVID, etc."
    )
    ap.add_argument(
        "--size",
        default=None,
        help="Force output size, e.g. 1280x720. If not set, use first image size.",
    )
    ap.add_argument(
        "--keep_aspect",
        action="store_true",
        help="If --size is set, keep aspect ratio with letterbox.",
    )
    ap.add_argument(
        "--repeat_last",
        type=int,
        default=0,
        help="Repeat last frame N times (useful to pause at end).",
    )
    ap.add_argument(
        "--start", type=int, default=0, help="Start index in sorted image list."
    )
    ap.add_argument(
        "--end", type=int, default=-1, help="End index (exclusive). -1 means all."
    )
    args = ap.parse_args()

    images = list_images(args.input_dir)
    if not images:
        raise RuntimeError(f"No images found in {args.input_dir}")

    # slice
    end = args.end if args.end != -1 else len(images)
    images = images[args.start : end]
    if not images:
        raise RuntimeError("After slicing, no images left. Check --start/--end.")

    first = read_image(images[0])
    if first is None:
        raise RuntimeError(f"Failed to read first image: {images[0]}")

    if args.size is None:
        h, w = first.shape[:2]
    else:
        w, h = parse_size(args.size)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(args.output, fourcc, float(args.fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            "Failed to open VideoWriter. Try a different --codec (e.g. mp4v/avc1) or output extension."
        )

    count = 0
    for p in images:
        img = read_image(p)
        if img is None:
            print(f"[WARN] skip unreadable: {p}")
            continue

        if args.size is not None:
            img = resize_to(img, w, h, keep_aspect=args.keep_aspect)
        else:
            ih, iw = img.shape[:2]
            if (iw, ih) != (w, h):
                # 如果输入图片尺寸不一致而你没指定 --size，这里强制拉伸到首帧尺寸
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        writer.write(img)
        count += 1

    # repeat last frame
    if args.repeat_last > 0 and count > 0:
        for _ in range(args.repeat_last):
            writer.write(img)

    writer.release()
    print(f"[OK] Wrote {count} frames -> {args.output} @ {args.fps} fps, size={w}x{h}")


if __name__ == "__main__":
    main()
