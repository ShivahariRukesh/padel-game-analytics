from ultralytics import YOLO
import cv2
import pickle


class PlayerTracker:

    def __init__(self,*, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames,masked_image_path, read_from_stub =False, stub_path=None):
        player_detections =[]

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        mask_img = cv2.imread(masked_image_path)
        
        for frame in frames:
            mask_img_resized = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]))

            # If mask is grayscale, convert to 3-channel
            if len(mask_img_resized.shape) == 2:
                mask_img_resized = cv2.cvtColor(mask_img_resized, cv2.COLOR_GRAY2BGR)

            imgRegion = cv2.bitwise_and(frame, mask_img_resized)

            # player_dict = self.detect_frame(frame)
            player_dict = self.detect_frame(imgRegion)

            player_detections.append(player_dict)



        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections,f)

        return player_detections
    
    def detect_frame(self,frame):
        # results = self.model.track(frame, persist=True)[0]
        results = self.model.track(
    frame,
    tracker='botsort.yaml',
    imgsz=640,
    conf=0.3,
    iou=0.5,
    persist=True,
)[0]
        id_name_dict = results.names 
        player_dict = {}

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_label = id_name_dict[object_cls_id]

            if object_cls_label == "person":
                player_dict[track_id] = result

            if object_cls_label == "tennis racket":
                player_dict["racket"] = result

        return player_dict
        

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []

        for frame, player_dict in zip(video_frames,player_detections):
            for track_id, bbox in player_dict.items():
                x1,y1,x2,y2 = bbox
                cv2.putText(frame, f"{'Racket' if track_id=='racket' else f'Player: {track_id}'}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 2)
                cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,255,0), 2 )
        
            output_video_frames.append(frame)

        return output_video_frames


