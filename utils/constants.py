# paths 
input_video_path        = "./input_videos/input_video.mp4"
output_video_path       = "./output_videos/output_video.avi"
masked_image_path       = "./input_videos/input_mask_image.png"

model_path_tracker      = "./yolo_model_weights/yolov8x.pt"
model_path_ball         = "./yolo_model_weights/last.pt"
model_path_shot_type    = "./yolo_model_weights/pose/last.pt"

stub_player             = "./tracker_stubs/player_detections_mask.pkl"
stub_ball               = "./tracker_stubs/ball_detections.pkl"
stub_shot_type          = "./tracker_stubs/shot_type_detections.pkl"


VELOCITY_SMOOTH_WINDOW   = 4    # frames averaged on each side for velocity
MIN_VELOCITY_PX          = 8    # ignore tiny movements (noise filter)
MIN_FRAMES_BETWEEN_HITS  = 18   # debounce: ignore strikes within this many frames
SHOT_LABEL_WINDOW        = 6    # frames to search around strike for a shot label
FLASH_DURATION_FRAMES    = 14   # how long the "FOREHAND / BACKHAND!" flash stays


