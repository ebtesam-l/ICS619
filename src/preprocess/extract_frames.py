# USAGE: python extract_frames.py --input video.mp4 --output frames/
# Or pass a folder of videos: python extract_frames.py --input videos_dir/ --output frames/

import argparse
import os
import glob
import cv2

FPS_OUT = 1  # one frame per second


def extract_frames_from_video(video_path, out_dir):
    """Read a video and save one frame per second under out_dir.

    Returns the number of frames written.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps / FPS_OUT)))
    base = os.path.splitext(os.path.basename(video_path))[0]

    idx, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out = os.path.join(out_dir, f"{base}_f{written:06d}.png")
            cv2.imwrite(out, frame)
            written += 1
        idx += 1
    cap.release()
    return written


def extract_frames(input_path, out_dir):
    """Dispatch over a single video or a folder of videos."""
    if os.path.isdir(input_path):
        videos = sorted(
            glob.glob(os.path.join(input_path, "*.mp4")) +
            glob.glob(os.path.join(input_path, "*.mov")) +
            glob.glob(os.path.join(input_path, "*.avi")) +
            glob.glob(os.path.join(input_path, "*.mkv"))
        )
    else:
        videos = [input_path]

    total = 0
    for v in videos:
        n = extract_frames_from_video(v, out_dir)
        print(f"  {os.path.basename(v)} -> {n} frames")
        total += n
    return total


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", help="video file or folder of videos", required=True)
    argParser.add_argument("--output", help="folder to write frames into", required=True)
    args = argParser.parse_args()

    n = extract_frames(args.input, args.output)
    print(f"Total frames written: {n}")
