import cv2
from ultralytics import YOLO
model = YOLO('./runs/detect/train/weights/last.pt')

image = cv2.imread('/Users/agrimsigdel/Downloads/ANPR/T.jpeg')
results = model(image)

boxes = results[0].boxes.xyxy.cpu().numpy()
for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#     run_yolo_np_recognition_on_video()
#     run_yolo_np_recognition_on_video(model_path="./runs/detect/train/weights/best.pt", video_source='./B.mov')
#     run_yolo_np_recognition_on_video(model_path="./runs/detect/train/weights/best.pt", video_source=0)