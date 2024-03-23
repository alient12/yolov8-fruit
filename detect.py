from ultralytics import YOLO
from PIL import Image

# Load a model
# model = YOLO('best.pt')  # load an official model
model = YOLO("C:/Users/AliEntezari/Desktop/Athena Codes/Academus_project/yolov8/runs/detect/train/weights/best.pt")  # load an official model

# # Predict with the model
# results = model(['test-640.jpg'])

# # Process results list
# i = 0
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     # masks = result.masks  # Masks object for segmentation masks outputs
#     # keypoints = result.keypoints  # Keypoints object for pose outputs
#     # probs = result.probs  # Probs object for classification outputs

#     im_array = result.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     # im.save(f'results_{i}.jpg')  # save image
#     # i -=- 1

# model.predict(['test.jpg', 'test2.jpg', 'test-640.jpg'], save=True, conf=0.85)
# model.predict('C:/Users/AliEntezari/Desktop/Athena Codes/Academus_project/datasets/Fruits fresh and rotten for classification/test/freshapples', save=True, conf=0.85)
# model.predict('C:/Users/AliEntezari/Desktop/Athena Codes/Academus_project/datasets/Fruits fresh and rotten for classification/test/rottenoranges', save=True, conf=0.85)
model.predict('C:/Users/AliEntezari/Desktop/Athena Codes/Academus_project/datasets/Fruits fresh and rotten for classification/valid/rottenapples', save=True, conf=0.85)