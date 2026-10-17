from ultralytics import YOLO
import cv2
model = YOLO("best.pt")
video_path = "armor.mp4"
output_video_path = "result.mp4"
cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 # 获取视频基础参数
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
     # YOLO推理
    results = model(frame)
     # 自动绘制检测框、类别、置信度、4个关键点
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

cap.release()
out.release()
print(f"视频推理完成，保存为 {output_video_path}")