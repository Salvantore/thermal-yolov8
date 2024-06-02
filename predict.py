from ultralytics import YOLO
from PIL import Image

model = YOLO('C:/Users/PC/Desktop/DATASET_THERMAL/runs/detect/train8/weights/best.pt')

results = model('image.mp4')  


for r in results:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1]) #RGB PIL image
    im.show()
    im.save('results_image_m.mp4')
