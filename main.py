from ultralytics import YOLO
from PIL import Image

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model(['cow2.jpg', 'cow.jpg', 'time-square2.jpg', 'time-square.jpg'])

# Process results list
i = 0
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs

    im_array = result.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    # im.save(f'results_{i}.jpg')  # save image
    # i -=- 1