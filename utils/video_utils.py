import os
import cv2


def read_video(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    cap.release()
    return frames


def iter_chunks(video_path: str, chunk_size: int = 64):
    """
    Yields (chunk_start_index, list_of_frames) in batches of `chunk_size`.
    At most `chunk_size` frames live in RAM at any moment.

    Usage:
        for start, frames in iter_chunks("video.mp4", chunk_size=64):
            detections = tracker.detect_frames(frames)
            ...
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    chunk_start = 0
    chunk: list = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        chunk.append(frame)
        if len(chunk) == chunk_size:
            yield chunk_start, chunk
            chunk_start += len(chunk)
            chunk = []

    if chunk:                  
        yield chunk_start, chunk

    cap.release()


def get_video_info(video_path: str) -> dict:
    """Return fps, total_frames, and true frame dimensions from the first decoded frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 24.0

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not decode first frame from: {video_path}")

    height, width = frame.shape[:2]
    return {"fps": fps, "total_frames": total_frames, "width": width, "height": height}


def save_video(output_video_frames: list, output_video_path: str, fps: float = 24.0) -> None:
    """
    Saves frames to disk.

    Tries codecs in order until one actually opens:
        mp4v + .mp4   (most portable)
        avc1 + .mp4   (H.264)
        XVID + .avi
        MJPG + .avi   (original — last resort, often missing on Linux)

    Raises RuntimeError if every codec fails so you get a clear message
    instead of a silent 24 KB file.
    """
    if not output_video_frames:
        raise ValueError("output_video_frames is empty — nothing to save.")

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)


    height, width = output_video_frames[0].shape[:2]
    frame_size    = (width, height)

    base, _ = os.path.splitext(output_video_path)

    candidates = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]

    writer = None
    chosen_path = None

    for fourcc_str, ext in candidates:
        path   = base + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        w      = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if w.isOpened():
            writer      = w
            chosen_path = path
            print(f"[save_video] codec={fourcc_str}  size={width}x{height}  -> {path}")
            break
        w.release()
        print(f"[save_video] codec {fourcc_str}{ext} unavailable, trying next…")

    if writer is None:
        raise RuntimeError(
            "cv2.VideoWriter failed with every codec.\n"
            "Fix: sudo apt-get install ffmpeg   (or)   "
            "pip install opencv-python-headless"
        )

    for frame in output_video_frames:
        writer.write(frame)

    writer.release()

    file_kb = os.path.getsize(chosen_path) / 1024
    print(f"[save_video] saved {len(output_video_frames)} frames  |  {file_kb:.1f} KB  ->  {chosen_path}")

    if file_kb < 50:
        print(
            "[save_video] WARNING: file is suspiciously small (<50 KB).\n"
            "             Codec may be present but broken. Try: sudo apt-get install ffmpeg"
        )