import requests
import base64
import numpy as np
from PIL import Image
import io
import os

def img_to_base64(img):
    pil_img = Image.fromarray(img.astype(np.uint8))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def main():
    # 假设已用 extract_frames.py 生成若干测试帧
    frame_dir = "frames_test"
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if len(frame_files) < 2:
        print("请先用 video_frame_extractor.py 生成至少两帧测试图片")
        return

    # 加载第一帧和初始化关键点
    first_img = np.array(Image.open(frame_files[0]))
    keypoints = np.array([[50, 100], [120, 200]], dtype=np.float32)  # [x, y] 格式

    # 1. 注册初始帧和关键点
    url = "http://localhost:5000/register"
    payload = {
        "frame": img_to_base64(first_img),
        "keypoints": keypoints.tolist()
    }
    resp = requests.post(url, json=payload)
    print("Register:", resp.status_code, resp.json())
    
    # 2. 依次跟踪后续帧
    url = "http://localhost:5000/track"
    for i, fname in enumerate(frame_files[1:]):
        img = np.array(Image.open(fname))
        payload = {
            "frame": img_to_base64(img)
        }
        resp = requests.post(url, json=payload)
        print(f"Track frame {i+2}:", resp.status_code, resp.json())

if __name__ == "__main__":
    main()