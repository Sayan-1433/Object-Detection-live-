from ultralytics import RTDETR
import cv2
import torch

# ----------------------------
# PERFORMANCE SETTINGS
# ----------------------------
INPUT_SIZE = (720, 640)   # smaller = faster
CONF_THRESH = 0.6         # higher = fewer boxes
DETECT_EVERY = 5          # run detection every N frames
SHOW_LABELS = True        # set False for more speed
CAMERA_INDEX = 0
# ----------------------------

# Limit CPU thread usage (often helps on laptops)
torch.set_num_threads(6) # as 12 is available

# Load RT-DETR (CPU-friendly)
model = RTDETR("rtdetr-l.pt")
model.model.eval()

# Open webcam
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ RT-DETR webcam running")
print("Press Q to quit")

frame_id = 0
last_results = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Resize early (major speed gain)
    frame = cv2.resize(frame, INPUT_SIZE)

    # Run detection every N frames
    if frame_id % DETECT_EVERY == 0:
        last_results = model(
            frame,
            conf=CONF_THRESH,
            verbose=False
        )

    # Draw last detections
    if last_results is not None:
        frame = last_results[0].plot(
            labels=SHOW_LABELS,
            boxes=True
        )

    cv2.imshow("RT-DETR Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
