import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Load pretrained DETR from Torch Hub (official)
model = torch.hub.load(
    'facebookresearch/detr',
    'detr_resnet50',
     pretrained=True
    )

model.eval()

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

transform = T.Compose([
    T.Resize(600), # to reduce hallucination
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open("test.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(tensor)

probs = outputs["pred_logits"].softmax(-1)[0, :, :-1]
boxes = outputs["pred_boxes"][0]

keep = probs.max(-1).values > 0.85

plt.imshow(img)
ax = plt.gca()
w, h = img.size

for p, box in zip(probs[keep], boxes[keep]):
    class_id = p.argmax().item()
    score = p.max().item()
    label = CLASSES[class_id]

    #if score < 0.9:
    #    continue

    cx, cy, bw, bh = box
    x1 = (cx - bw/2) * w
    y1 = (cy - bh/2) * h

    ax.add_patch(
        plt.Rectangle((x1, y1), bw*w, bh*h,
                      fill=False, color='red', linewidth=2)
    )

    ax.text(
        x1, y1,
        f"{label}: {score:.2f}",
        color='white',
        bbox=dict(facecolor='red', alpha=0.7)
    )

plt.axis("off")
plt.show()
