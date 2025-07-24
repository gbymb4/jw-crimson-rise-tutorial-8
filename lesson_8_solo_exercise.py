# -*- coding: utf-8 -*-  
"""  
Ultra-Fast Laptop Object Detection Mini-Project  
SCAFFOLD: Student implements core model and inference routines.  
  
- Data: Synthetic coco-like, 3 classes, 60 total samples.  
- Task 1: Implement get_model(), experiment with backbone, #classes etc.  
- Task 2: Implement detect() function, given an image tensor.  
- Task 3: Implement compare_confidence_levels() to analyze detection counts.  
- All training, evaluation, and visualization is provided.  
"""  
  
import torch  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
from torchvision.models.detection import (  
    fasterrcnn_mobilenet_v3_large_fpn,  
    fasterrcnn_resnet50_fpn  
)  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  
import numpy as np  
import matplotlib.pyplot as plt  
import random  
  
# ---- Synthetic Data Setup (tiny "COCO") ----  
  
CLASSES = ["person", "car", "bottle"]  
IDX_TO_CLASS = {i + 1: c for i, c in enumerate(CLASSES)}  
NUM_CLASSES = len(CLASSES) + 1  # +1 for background  
  
def mini_coco(num_images=60, size=224):  
    rng = np.random.RandomState(0)  
    data = []  
    for i in range(num_images):  
        img = torch.rand(3, size, size) * 0.2 + 0.7  
        nobj = rng.randint(1, 3)  
        boxes, labels = [], []  
        for _ in range(nobj):  
            w, h = rng.randint(32, 64), rng.randint(32, 64)  
            x1 = rng.randint(0, size - w)  
            y1 = rng.randint(0, size - h)  
            x2, y2 = x1 + w, y1 + h  
            label = rng.randint(1, len(CLASSES) + 1)  
            img[:, y1:y2, x1:x2] = (label / (len(CLASSES) + 2))  
            boxes.append([x1, y1, x2, y2])  
            labels.append(label)  
        data.append({"image": img, "boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels)})  
    return data  
  
class TinyCoco(Dataset):  
    def __init__(self, split="train", total=60):  
        self.data = mini_coco(int(total * 0.7) if split == "train" else int(total * 0.3))  
    def __getitem__(self, idx):  
        d = self.data[idx]  
        t = {"boxes": d["boxes"], "labels": d["labels"], "image_id": torch.tensor([idx])}  
        return d["image"], t  
    def __len__(self): return len(self.data)  
  
def collate_fn(batch): return tuple(zip(*batch))  
  
# ---- EXERCISE: STUDENT FUNCTIONS ----  
  
def get_model():  
    """  
    Returns a detection model for NUM_CLASSES classes using a Torchvision backbone.  
  
    STUDENT TASK:  
    - Optionally switch MobileNet to ResNet50 (see below).  
    - Ensure the model head outputs NUM_CLASSES (=4) [3 classes + background].  
  
    Example:  
        model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")  
        # model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  
        # Replace prediction head for NUM_CLASSES  
        in_features = model.roi_heads.box_predictor.cls_score.in_features  
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)  
  
    Return:  
        A ready-to-train model.  
    """  
    # --- STUDENT: EDIT BELOW ---  
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")  
    # model = fasterrcnn_resnet50_fpn(weights="DEFAULT")   # try this for more accuracy (slower!)  
    in_features = model.roi_heads.box_predictor.cls_score.in_features  
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)  
    return model  
  
  
def detect(model, image, device, score_thresh=0.3):  
    """  
    Runs model inference on a single tensor image.  
    Returns boxes, labels, scores (numpy) of detections above score_thresh.  
  
    Pseudocode:  
    1. Set model to eval mode.  
    2. Move image tensor to the correct device.  
    3. Pass [image] (as a list) through the model with torch.no_grad().  
    4. Get the first prediction dict from output list.  
    5. Extract and convert 'boxes', 'labels', 'scores' to numpy arrays.  
    6. Select entries where score > score_thresh.  
    7. Return filtered boxes, labels, scores.  
    """  
    raise NotImplementedError("Implement this function as your exercise!")  
  
  
def compare_confidence_levels(model, image, device, conf_thresholds=[0.2, 0.4, 0.6, 0.8]):  
    """  
    For each threshold, runs detect() and prints number of detections above that confidence.  
  
    Pseudocode:  
    For each threshold in conf_thresholds:  
        1. Call detect(model, image, device, score_thresh=threshold)  
        2. Count number of boxes returned.  
        3. Print threshold and number of detections.  
    """  
    for thresh in conf_thresholds:  
        print(f"conf {thresh:.2f}: ... (student implements result)")  
  
# ---- Training, Visualization (don't edit!) ----  
  
def train_one_epoch(model, loader, optimizer, device):  
    model.train()  
    avg_loss = 0  
    for imgs, targets in loader:  
        imgs = [i.to(device) for i in imgs]  
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  
        optimizer.zero_grad()  
        loss_dict = model(imgs, targets)  
        loss = sum(loss for loss in loss_dict.values())  
        loss.backward()  
        optimizer.step()  
        avg_loss += loss.item()  
    return avg_loss / len(loader)  
  
def quick_visualize(model, loader, device, detect_fn, n=3):  
    model.eval()  
    fig, axes = plt.subplots(1, n, figsize=(15, 5))  
    if n == 1: axes = [axes]  
    with torch.no_grad():  
        for i, (imgs, tars) in enumerate(loader):  
            if i >= n: break  
            img = imgs[0]  
            gt = tars[0]  
            npimg = img.permute(1, 2, 0).cpu().numpy().clip(0, 1)  
            ax = axes[i]  
            ax.imshow(npimg)  
            # Draw GT boxes  
            for b, l in zip(gt["boxes"], gt["labels"]):  
                x1, y1, x2, y2 = b.cpu()  
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=0, color="green", lw=2))  
                ax.text(x1, y1-5, f"GT:{IDX_TO_CLASS[l.item()]}", color='green', fontsize=8)  
            # Draw model predictions (calls student's detect)  
            try:  
                boxes, labels, scores = detect_fn(model, img, device)  
                for b, l, s in zip(boxes, labels, scores):  
                    x1, y1, x2, y2 = b  
                    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=0, color="red", lw=2))  
                    ax.text(x1, y2+4, f"{IDX_TO_CLASS.get(int(l),'unk')}:{s:.2f}", color='red', fontsize=8)  
            except NotImplementedError:  
                pass  
            ax.set_axis_off()  
            ax.set_title(f"Sample {i+1}")  
    plt.tight_layout(); plt.show()  
  
# ---- Main Routine ----  
  
def main():  
    torch.manual_seed(1); np.random.seed(1); random.seed(1)  
    device = torch.device("cpu")  
    print(f"DEVICE={device}")  
  
    # Data loading  
    train_ds = TinyCoco("train")  
    test_ds = TinyCoco("test")  
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)  
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)  
  
    # Get student's model definition  
    model = get_model().to(device)  
    optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9)  
  
    print("--> Training (very fast, tiny data)")  
    losses = []  
    for e in range(3):  
        loss = train_one_epoch(model, train_dl, optimizer, device)  
        print(f"Epoch {e+1}: loss={loss:.4f}")  
        losses.append(loss)  
  
    plt.figure(); plt.plot(losses, marker='o'); plt.title("Loss"); plt.xlabel("Epoch"); plt.show()  
  
    print("--> Visualization (prediction will appear after detect() is implemented)")  
    quick_visualize(model, test_dl, device, detect, n=3)  
  
    # Student Exercise: Confidence sweep, after detect() implemented  
    # Example usage:  
    img, _ = test_ds[0]  
    compare_confidence_levels(model, img, device)  
  
    print("\nDone! Now edit get_model(), detect(), and compare_confidence_levels() as indicated in the docstrings to finish the exercise.")  
  
if __name__ == "__main__":  
    import warnings  
    warnings.filterwarnings("ignore")  
    main()  