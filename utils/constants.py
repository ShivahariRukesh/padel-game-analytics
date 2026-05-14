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


VELOCITY_SMOOTH_WINDOW   = 4    
MIN_VELOCITY_PX          = 8    
MIN_FRAMES_BETWEEN_HITS  = 18   
SHOT_LABEL_WINDOW        = 6    
FLASH_DURATION_FRAMES    = 14   