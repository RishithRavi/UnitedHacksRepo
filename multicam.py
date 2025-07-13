from ultralytics import YOLO
import cv2
import math
import redis
import json
import threading
import time

# ---- CONFIG ----
camera_sources = [0]  # Change if using RTSP cameras
CONFIDENCE_THRESHOLD = 0.5
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# Load both YOLO models
print("✅ Loading YOLO models...")
gun_model = YOLO("best.pt")  # Gun detection model
person_model = YOLO("yolov8n.pt")  # Person detection model

# Class Names
gun_class_names = ["Firearm"]  # Gun detection model classes
person_class_names = ["Person"]  # Person detection model classes

# Connect to Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)


# ---- FUNCTION: Store Flagged Person in Redis ----
def store_flagged_person(person_id, camera_id):
    data = {"person_id": person_id, "camera_id": camera_id, "timestamp": time.time()}
    redis_client.set(f"flagged_person:{person_id}", json.dumps(data))


# ---- FUNCTION: Retrieve Flagged Person ----
def get_flagged_person(person_id):
    data = redis_client.get(f"flagged_person:{person_id}")
    return json.loads(data) if data else None


# ---- FUNCTION: Draw Bounding Boxes ----
def draw_boxes(img, boxes, color, class_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
        conf = float(box.conf[0])  # Confidence score
        cls = int(box.cls[0])  # Class index

        if conf < CONFIDENCE_THRESHOLD:
            continue  # Skip low-confidence detections

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Add class label
        if cls < len(class_names):
            label = f"{class_names[cls]} ({conf:.2f})"
            cv2.putText(
                img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )


# ---- FUNCTION: Process Camera Feed ----
def process_camera(source, index):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ ERROR: Camera {index} not opening.")
        return

    print(f"✅ Camera {index} started...")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print(f"❌ Failed to grab frame from Camera {index}.")
            break

        # ---- RUN BOTH MODELS SIMULTANEOUSLY ----
        gun_results = gun_model(img)
        person_results = person_model(img)

        # Extract detections
        gun_boxes = [
            box for result in gun_results for box in result.boxes if box.cls == 0
        ]  # Guns
        person_boxes = [
            box for result in person_results for box in result.boxes if box.cls == 0
        ]  # People

        # ---- DRAW BOUNDING BOXES ----
        draw_boxes(img, gun_boxes, (0, 0, 255), gun_class_names)  # Red for guns
        draw_boxes(
            img, person_boxes, (0, 255, 0), person_class_names
        )  # Green for people

        # ---- DISPLAY FRAME ----
        cv2.imshow(f"Camera {index}", img)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---- START CAMERA FEED ----
threads = []
for i, source in enumerate(camera_sources):
    thread = threading.Thread(target=process_camera, args=(source, i))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
