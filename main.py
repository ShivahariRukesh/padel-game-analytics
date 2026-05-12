
import os
import cv2
import pickle

from utils import iter_chunks, get_video_info
from detectors import PlayerTracker, BallTracker, ShotTypeDetector, ShotCounter



INPUT_VIDEO_PATH  = "input_videos/input_sample_video.mp4"
OUTPUT_VIDEO_PATH = "output_videos/output_sample_video_2.mp4"
MASKED_IMAGE_PATH = "input_videos/input_cut_mask_image.png"

MODEL_TRACKER   = "yolo_model_weights/yolov8x.pt"
MODEL_BALL      = "yolo_model_weights/last.pt"
MODEL_SHOT_TYPE = "yolo_model_weights/pose/best.pt"

STUB_PLAYER    = "tracker_stubs/player_detections_sample_2.pkl"
STUB_BALL      = "tracker_stubs/ball_detections_sample_2.pkl"
STUB_SHOT_TYPE = "tracker_stubs/shot_type_detections_sample_2.pkl"




CHUNK_SIZE              = 64
OUTPUT_FPS              = 24.0
READ_PLAYER_FROM_STUB   = False
READ_BALL_FROM_STUB     = False
READ_SHOTTYPE_FROM_STUB = False



def _load_stub(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _save_stub(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)



def _run_detection_pass(tracker, stub_path, read_from_stub, video_path, **detect_kwargs):
    if read_from_stub and os.path.exists(stub_path):
        print(f"  loading stub: {stub_path}")
        return _load_stub(stub_path)

    all_detections = []
    for chunk_start, frames in iter_chunks(video_path, CHUNK_SIZE):
        dets = tracker.detect_frames(frames, **detect_kwargs)
        all_detections.extend(dets)
        print(f"  {tracker.__class__.__name__}: processed up to frame {chunk_start + len(frames)}")

    _save_stub(all_detections, stub_path)
    return all_detections



def main():
    info = get_video_info(INPUT_VIDEO_PATH)
    print(
        f"\nVideo: {info['total_frames']} frames | "
        f"{info['fps']:.1f} fps | "
        f"{info['width']}x{info['height']}"
    )

    
    player_tracker     = PlayerTracker(model_path=MODEL_TRACKER)
    ball_tracker       = BallTracker(MODEL_BALL)
    shot_type_detector = ShotTypeDetector(MODEL_SHOT_TYPE)

    
    print("\n=== PASS 1: Player detection ===")
    player_detections = _run_detection_pass(
        player_tracker, STUB_PLAYER, READ_PLAYER_FROM_STUB,
        INPUT_VIDEO_PATH,
        masked_image_path=MASKED_IMAGE_PATH,  
    )

    print("\n=== PASS 1: Ball detection ===")
    ball_detections = _run_detection_pass(
        ball_tracker, STUB_BALL, READ_BALL_FROM_STUB,
        INPUT_VIDEO_PATH,
    )

    print("\n=== PASS 1: Shot-type detection ===")
    shot_type_detections = _run_detection_pass(
        shot_type_detector, STUB_SHOT_TYPE, READ_SHOTTYPE_FROM_STUB,
        INPUT_VIDEO_PATH,
    )

    
    print("\n=== Shot analysis ===")
    shot_counter = ShotCounter(fps=OUTPUT_FPS)
    shot_counter.analyze(ball_detections, player_detections, shot_type_detections)
    shot_counter.export_json("output_data/shot_log.json")
    shot_counter.export_csv("output_data/shot_log.csv")

    
    print("\n=== PASS 2: Drawing & saving ===")

    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH) or ".", exist_ok=True)
    frame_size = (info["width"], info["height"])
    base, _    = os.path.splitext(OUTPUT_VIDEO_PATH)

    writer      = None
    chosen_path = None

    for fourcc_str, ext in [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]:
        path   = base + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        w      = cv2.VideoWriter(path, fourcc, OUTPUT_FPS, frame_size)
        if w.isOpened():
            writer      = w
            chosen_path = path
            print(f"  codec={fourcc_str}  output={path}")
            break
        w.release()
        print(f"  codec {fourcc_str} unavailable, trying next...")

    if writer is None:
        raise RuntimeError(
            "cv2.VideoWriter failed with every codec.\n"
            "Run:  sudo apt-get install ffmpeg"
        )

    frames_written = 0

    for chunk_start, frames in iter_chunks(INPUT_VIDEO_PATH, CHUNK_SIZE):
        chunk_end = chunk_start + len(frames)

        p_chunk = player_detections   [chunk_start:chunk_end]
        b_chunk = ball_detections     [chunk_start:chunk_end]
        

        out = player_tracker    .draw_bboxes(frames, p_chunk)
        out = ball_tracker      .draw_bboxes(out,    b_chunk)
        

        for local_i, frame in enumerate(out):
            shot_counter.draw_overlay(frame, chunk_start + local_i)

        
        if frames_written == 0:
            h, w = out[0].shape[:2]
            if (w, h) != frame_size:
                writer.release()
                raise ValueError(
                    f"Frame size mismatch: writer={frame_size}, frame={w}x{h}.\n"
                    "A draw_bboxes call is resizing frames -- check your trackers."
                )

        for frame in out:
            writer.write(frame)
        frames_written += len(out)
        print(f"  wrote frames {chunk_start}-{chunk_end - 1}  (total: {frames_written})")

    writer.release()

    file_kb = os.path.getsize(chosen_path) / 1024
    print(f"\n=== Done ===")
    print(f"  frames written : {frames_written}")
    print(f"  file size      : {file_kb:.1f} KB")
    print(f"  output         : {chosen_path}")

    if file_kb < 50:
        print("  WARNING: file looks too small. Run: sudo apt-get install ffmpeg")


if __name__ == "__main__":
    main()