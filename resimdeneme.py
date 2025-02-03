import os
import cv2
from ultralytics import YOLO

class ParkingLotDetector:
    def __init__(self, model_path):

        self.model = YOLO(model_path)
        self.classes = ['vacant', 'occupied']

    def process_image(self, image_path, scale_factor=1.5):

        if not os.path.exists(image_path):
            print(f"Dosya bulunamadı: {image_path}")
            return

        frame = cv2.imread(image_path)
        if frame is None:
            print("Görüntü yüklenemedi.")
            return


        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)


        results = self.model(frame)
        detections = results[0].boxes

        vacant_count = 0
        occupied_count = 0

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            # Sınıf ismini al
            class_name = self.classes[class_id]

            if class_name == 'vacant':
                color = (0, 255, 0)
                vacant_count += 1
            else:
                color = (0, 0, 255)
                occupied_count += 1


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"Vacant: {vacant_count}, Occupied: {occupied_count}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


        cv2.namedWindow("Parking Lot Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Parking Lot Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


        cv2.imshow("Parking Lot Detection", frame)


        cv2.waitKey(0)
        cv2.destroyAllWindows()




model_path = os.path.join(os.getcwd(), "best (3).pt")
image_path = os.path.join(os.getcwd(), "resim", "araba2.jpeg")


detector = ParkingLotDetector(model_path)
detector.process_image(image_path)
