from ultralytics import RTDETR
import cv2

# Load pretrained RT-DETR (smallest, CPU-friendly)
model = RTDETR("rtdetr-l.pt")  # auto-downloads weights

# Load image
img = cv2.imread("test.jpg")

# Run inference
results = model(img, conf=0.5)

# Visualize results
annotated = results[0].plot()

cv2.imshow("RT-DETR Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
