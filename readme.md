# Padel Game Analytics — Shot Classification System

A computer vision system for detecting and tracking tennis players, tennis rackets, and the ball from a video using YOLOv8 and OpenCV.
The system processes an input tennis video, applies a mask to focus on the court region, tracks players using YOLO tracking, detects the tennis ball using a custom-trained YOLO model, and saves the annotated output video.


## How it works
 
**1. Load video metadata**
`get_video_info` opens the input video and reads the FPS, total frame count, and true decoded resolution. This is used to size the output writer and timestamp every detected shot correctly.
 
**2. Detect players and rackets**
The input video is read in 64-frame chunks. Each chunk is masked with a court image (`input_cut_mask_image.png`) so only pixels inside the play area reach the model. YOLOv8x with BotSort persistent tracking is then run on each masked frame, returning a bounding box and a stable numeric `track_id` for every player, plus a separate `"racket"` entry when a tennis racket is visible.
 
**3. Detect the ball**
The same 64-frame chunks are passed to a custom-trained YOLO ball detector. It runs at a low confidence threshold (0.15) to maximise recall on motion-blurred or partially occluded frames, returning a bounding box for the ball in each frame where one is found.
 
**4. Detect shot type**
A pose-based YOLO model inspects each frame and classifies the visible player posture as either `forehand` or `backhand`, returning the class name and bounding box per frame.
 
**5. Cache all detections**
After each pass the full list of per-frame detections is saved to a `.pkl` stub file. On subsequent runs you can set `READ_*_FROM_STUB = True` to skip inference entirely and load the cached results instead.
 
**6. Analyse shots (ShotCounter)**
`ShotCounter.analyze` fuses the three detection lists into a shot log:
- Missing ball positions are filled in by linear interpolation.
- A smoothed horizontal velocity signal is computed for the ball's trajectory.
- Every point where the velocity changes sign (ball reverses horizontal direction) is treated as a candidate hit.
- Each candidate is confirmed by finding a matching `forehand` or `backhand` label within ±6 frames.
- The closest player to the ball at that moment is assigned as the hitter.
- Running forehand and backhand totals are tracked globally and per player.
**7. Export shot data**
The confirmed shot log is written to `output_data/shot_log.json` (with a summary block) and `output_data/shot_log.csv` (one row per shot).
 
**8. Draw and save the output video**
The input video is read in chunks one final time. For each frame, player and ball bounding boxes are drawn, and `ShotCounter.draw_overlay` stamps a semi-transparent HUD in the top-left corner showing the live forehand and backhand counts. The annotated frames are written to the output video file, trying four codec/container combinations (`mp4v`, `avc1`, `XVID`, `MJPG`) until one succeeds.


## Project Structure

```
padel-game-analytics/
│
├── main.py                        # Entry point — orchestrates the full pipeline
│
├── detectors/
│   ├── __init__.py
│   ├── player_tracker.py          # PlayerTracker  — detect + track players & racket
│   ├── ball_detector.py           # BallTracker    — detect ball per frame
│   ├── shot_counter.py            # ShotCounter — strike detection, HUD, export
|   └── shot_type_detector.py      # ShotTypeDetector — classify forehand / backhand
│
├── model_training/
│   ├── player_shot_pose_training.ipynb
│   └── tennis_ball_detector_training.ipynb 
|
├── utils/
│   ├── __init__.py
│   ├── constants.py
│   └── video_utils.py             # read_video / save_video helpers
│
├── yolo_model_weights/
|   | 
│   ├── ball_model/                # Fine-tuned ball detection model
│   |   |── best.pt  
│   |   └── last.pt 
|   | 
│   └── detect_model/              # General COCO model for player/racket detection
│   |   └── yolov8x.pt
|   |          
│   └── shot_model/                # Fine-tuned shot-type / pose classification model
│       |── best.pt  
│       └── last.pt              
|                   
|               
│
├── input_videos/
│   ├── input_video.mp4            # Source tennis footage
│   └── input_mask_image.png       # Binary mask — white = valid court area
│
├── output_videos/
│   └── output_video.mp4           # Annotated output video
│
├── output_data/
│   ├── shot_log.json              # Structured shot results (JSON)
│   └── shot_log.csv               # Structured shot results (CSV)
│
└── tracker_stubs/                 # Cached pickle files for fast re-runs
    ├── player_detection_stub.pkl
    ├── ball_detection_stub.pkl
    └── shot_type_detection.pkl
```



## Getting Started

### - Python Version
- Python version 3.10.14 is used for this project.
- A Python Virtual Environment is created to isolate project dependencies and manage different package or Python versions separately for this project. Using this command :
` python -m venv <name_of the environment> `, eg : ` python -m venv venv310 `


### - Install dependencies using requirements.txt:
Here requirements.txt is a file that contains the required packages to run this project.
- pip install -r requirements.txt 
 

### - Training models
- Here 'model_training/player_shot_pose_training.ipynb' and 'model_training/tennis_ball_detector_training.ipynb' contains the code to pre-train the models for better detections of player shot type and tennis ball across the frames of a video.


### - Faster re-runs with stubs
On first run, set read_from_stub=False for all trackers.
On subsequent runs, flip to read_from_stub=True to skip re-inference and load cached detections instantly.


### - Running the Project
`python main.py`
- It generates output video inside output_videos/ and **json**, **csv** files inside output_data/ 


## Output Format
 
### JSON — `output_data/shot_log.json`
 
```json
{
  "summary": {
    "total_shots": <number>,
    "forehand_total": <number>,
    "backhand_total": <number>,
    "per_player": {
      "<player_id>": { "<forehand/backhand>": <number>, "backhand": <number> },
      "<player_id": { "<forehand/backhand>": <number>, "backhand": <number> },
       ....
    }
  },
  "shots": [
    { "frame": <number>,  "timestamp": <number>, "shot_type": "<forehand/backhand>", "player_id": <number> },
    { "frame": <number>,  "timestamp": <number>, "shot_type": "<forehand/backhand>", "player_id": <number> },
    ....
  ]
}
```
 
### CSV — `output_data/shot_log.csv`
 
```
frame,timestamp,shot_type,player_id
....., ........, ......., .........
```

## Module reference
 
### `PlayerTracker`
Wraps YOLOv8x with BotSort persistent tracking. Applies a binary court mask before inference to suppress detections outside the play area. Tracks both `person` (assigned a numeric `track_id`) and `tennis racket` (stored under the key `"racket"`).
 
### `BallTracker`
Lightweight YOLO wrapper tuned for small, fast-moving objects. Uses a low confidence threshold (`0.15`) to improve recall on motion-blurred frames.
 
### `ShotTypeDetector`
Pose-based YOLO model that classifies the player's body posture as `forehand` or `backhand`. Only the class name and bounding box are retained per frame.
 
### `ShotCounter`
Fuses ball trajectory with shot-type labels to detect strokes:
1. Interpolates missing ball positions linearly.
2. Computes a smoothed horizontal velocity signal.
3. Detects velocity-sign reversals (the ball changes direction → a hit occurred).
4. Associates each hit with the nearest shot-type label (within `±SHOT_LABEL_WINDOW` frames) and the closest player.
5. Renders a semi-transparent HUD scoreboard onto each frame.
### `video_utils`
- `iter_chunks` — memory-efficient frame iterator; yields `(start_index, frames)` batches.
- `get_video_info` — returns FPS, total frame count, and true decoded resolution.
- `save_video` — tries four codec/container combinations (`mp4v`, `avc1`, `XVID`, `MJPG`) and raises a clear error if none work.


### Video — `output_videos/output_video.mp4`
`An annotated video of the input video that was provided in the model.`

## Challenges Faced
- Video Masking : Masking the video frame was challenging as the detected object would overlap with the masked backgroud.

- Ball occlusion : when the ball is hidden behind a player for many consecutive frames, interpolation degrades and a strike may be missed or slightly mis-timed.

- Player Id swap : player IDs are assigned by YOLOv8's tracker and can swap if two players cross paths near the camera edge.

- Net shots / serves : very short or vertical trajectories have low horizontal velocity and may fall below MIN_VELOCITY_PX; lower the threshold or add a vertical-velocity check to catch these.

- Multiple balls : As there were multiple balls, only one was in motion as it was only used while others were lying and it became hinderance to the ball movement detection. 

## Future Improvements

- Add more shot types (serve, volley, smash) : Currently, its only detecting and analyzing two types of shots, backhand and forehand. So, in future will be adding different types of shots like volley, smash, serve.

- Finetuning and Hypertuning the model : As, the models for player tracking and shot type detection is not yet close to the required accuracy. The parameters of these models needed to be experimented more and the model can be tuned perfectly with right and adequate amount of datasets. 

