import cv2
import gradio as gr
import numpy as np

# 加载OpenCV自带的人脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    print("收到图片，类型：", type(image))
    if image is None:
        print("图片为 None")
        return np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        import PIL.Image
        if isinstance(image, np.ndarray):
            arr = image
        elif isinstance(image, PIL.Image.Image):
            arr = np.array(image)
        else:
            print("图片类型无法处理")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        print("图片 shape:", arr.shape)
        if arr.shape[-1] == 4:  # RGBA 转 RGB
            arr = arr[..., :3]
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
        print("检测到人脸数量：", len(faces))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("Error:", e)
        return np.zeros((480, 640, 3), dtype=np.uint8)

iface = gr.Interface(
    fn=detect_faces,
    inputs=gr.Image(label="上传图片或摄像头", sources=["upload", "webcam"], type="pil"),
    outputs=gr.Image(label="检测结果"),
    title="实时人脸检测演示",
    description="可上传图片或使用摄像头进行人脸检测"
)

iface.launch(debug=True)