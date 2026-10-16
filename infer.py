from ultralytics import YOLOimport cv2
 model = YOLO("yolov8n.pt")
 img_path = "test.jpg"
 results = model(img_path)
 result_img = results[0].plot()
 cv2.imwrite("result.jpg", result_img)
 print("推理完成，输出图片 result.jpg")
