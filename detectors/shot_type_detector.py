from ultralytics import YOLO
import cv2
import pickle

class ShotTypeDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub =False, stub_path=None):
        shot_type_detection =[]

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                shot_type_detection = pickle.load(f)
            return shot_type_detection

        for frame in frames:
            shot_types_dict = self.detect_frame(frame)
            shot_type_detection.append(shot_types_dict)


        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(shot_type_detection,f)

        return shot_type_detection
    
    def detect_frame(self,frame):
        results = self.model.predict(frame)[0]
        shot_types_dict ={}
        for box in results.boxes:
            cls_id = int(box.cls[0])

            class_name = self.model.names[cls_id]
            confidence = float(box.conf[0])
            print(class_name, confidence)

            result = box.xyxy.tolist()[0]
            shot_types_dict[class_name] = result

        return shot_types_dict
        

    # def draw_bboxes(self, video_frames, shot_type_detection):
    #     output_video_frames = []

    #     for frame, shot_types_dict in zip(video_frames,shot_type_detection):
    #         for shot_type, bbox in shot_types_dict.items():
    #             x1,y1,x2,y2 = bbox
    #             cv2.putText(frame, f"{shot_type}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,0,0), 2)
    #             cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 2 )
            
    #         output_video_frames.append(frame)

    #     return output_video_frames