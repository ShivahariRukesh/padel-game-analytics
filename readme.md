# Padel Game Analytics — Shot Classification System

A computer vision system for detecting and tracking tennis players, tennis rackets, and the ball from a video using YOLOv8 and OpenCV.
The system processes an input tennis video, applies a mask to focus on the court region, tracks players using YOLO tracking, detects the tennis ball using a custom-trained YOLO model, and saves the annotated output video.





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


# How It Works

## Step 1 — Read Video Frames

The video is loaded frame-by-frame using utility functions.

```python
video_frames = read_video(input_video_path)
```

---

## Step 2 — Apply Court Mask

A mask image is applied to remove unnecessary background regions.

```
imgRegion = cv2.bitwise_and(frame, mask_img_resized)
```

This improves player detection accuracy.

---

## Step 3 — Detect & Track Players

Players are detected using:

```
self.model.track(frame, persist=True)
```

Each player receives a unique tracking ID.

Example:

```text
Player: 1
Player: 2
```

---

## Step 4 — Detect Tennis Ball

The ball detector predicts bounding boxes for the tennis ball.

```
self.model.predict(frame, conf=0.15)
```

---

## Step 5 — Draw Bounding Boxes

Bounding boxes and labels are rendered on each frame using OpenCV.

Example labels:

```
Player: 1
Racket
Ball
```

---

## Step 6 — Save Output Video

Annotated frames are saved as a video.

```python
save_video(output_video_frames, output_video_path)
```

---
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

