import requests
import base64
import numpy as np
from PIL import Image
import io
import os
import cv2
import matplotlib.pyplot as plt

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
    keypoints = np.array([[475, 177], [469, 237]], dtype=np.float32)  # [x, y] 格式

    # 1. 注册初始帧和关键点
    url = "http://localhost:5000/register"
    payload = {
        "frame": img_to_base64(first_img),
        "keypoints": keypoints.tolist()
    }
    resp = requests.post(url, json=payload)
    print("Register:", resp.status_code, resp.json())
    
    # 保存初始帧和关键点
    output_dir = 'output_frames'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 在初始图像上绘制关键点
    first_img_with_keypoints = cv2.cvtColor(first_img, cv2.COLOR_RGB2BGR)
    for point in keypoints:
        x, y = point.astype(int)
        cv2.circle(first_img_with_keypoints, (x, y), 5, (0, 255, 0), -1)
    
    # 保存初始图像
    initial_output_path = os.path.join(output_dir, 'frame_1.jpg')
    cv2.imwrite(initial_output_path, first_img_with_keypoints)
    print(f'Initial frame saved to {initial_output_path}')
    
    # 2. 依次跟踪后续帧
    url = "http://localhost:5000/track"
    for i, fname in enumerate(frame_files[1:]):
        img = np.array(Image.open(fname))
        payload = {
            "frame": img_to_base64(img)
        }
        resp = requests.post(url, json=payload)
        print(f"Track frame {i+2}:", resp.status_code, resp.json())

        if resp.status_code == 200:
                # 假设响应中包含跟踪到的关键点
                keypoints = np.array(resp.json().get('keypoints', []))
                if i + 2 < 17:
                    print(f'Frame {i+2} 未正确跟踪，未返回 keypoints。')
                    if len(keypoints) == 0:
                        continue
                # 在图像上绘制关键点
                img_with_keypoints = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for point in keypoints:
                    x, y = point.astype(int)
                    cv2.circle(img_with_keypoints, (x, y), 5, (0, 255, 0), -1)
                # 保存图像
                output_path = os.path.join(output_dir, f'frame_{i+2}.jpg')
                cv2.imwrite(output_path, img_with_keypoints)
                print(f'Frame {i+2} saved to {output_path}')

if __name__ == "__main__":
    main()