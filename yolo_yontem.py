import os
import cv2
from ultralytics import YOLO

class ParkingLotDetector:
    def __init__(self, model_path):

        self.model = YOLO(model_path)
        self.classes = ['vacant', 'occupied']

    def process_video(self, video_path):


        if not os.path.exists(video_path):
            print(f"Dosya bulunamadı: {video_path}")
            return


        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Video açılamadı.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv8 tahmini
            results = self.model(frame)
            detections = results[0].boxes

            vacant_count = 0
            occupied_count = 0

            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])


                class_name = self.classes[class_id]

                if class_name == 'vacant':
                    color = (0, 255, 0)
                    vacant_count += 1
                else:
                    color = (0, 0, 255)
                    occupied_count += 1


                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            cv2.putText(frame, f"Vacant: {vacant_count}, Occupied: {occupied_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Parking Lot Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

model_path = os.path.join(os.getcwd(), "best (3).pt")
video_path = os.path.join(os.getcwd(), "resim", "carPark.mp4")

detector = ParkingLotDetector(model_path)
detector.process_video(video_path)

cv2.waitKey(0)
cv2.destroyAllWindows()
