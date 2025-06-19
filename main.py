from flask import Flask, request, jsonify
from tracker import CoPointTracker
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)
# 初始化跟踪器
tracker = CoPointTracker(window_size=5)

@app.route('/register', methods=['POST'])
def register():
    """
    注册初始关键点和视频的第一帧。
    这会调用 tracker.reset() 来初始化 CoTracker 模型。
    """
    if not request.json or 'frame' not in request.json or 'keypoints' not in request.json:
        return jsonify({"error": "Missing frame or keypoints in request"}), 400

    try:
        data = request.json
        
        # 解码base64图像并转换为RGB numpy array
        img_data = base64.b64decode(data['frame'])
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        frame = np.array(image)
        
        # 将关键点转换为numpy数组 (N, 2), [x, y]
        keypoints = np.array(data['keypoints'], dtype=np.float32)
        
        # 使用第一帧和关键点初始化或重置跟踪器
        tracker.reset(frame, keypoints)
        
        return jsonify({"message": "Tracker registered successfully with initial frame and keypoints."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/track', methods=['POST'])
def track_route():
    """
    在新的图像帧中跟踪关键点。
    每次调用都会向CoTracker模型发送新的一帧。
    """
    if not request.json or 'frame' not in request.json:
        return jsonify({"error": "Missing frame in request"}), 400
        
    try:
        data = request.json
        
        # 解码图像
        img_data = base64.b64decode(data['frame'])
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        frame = np.array(image)

        # 使用标准 step_track 流程：自动滑窗并返回最新一帧跟踪结果
        new_keypoints, visibility = tracker.step_track(frame)
        
        # 返回新的关键点位置和它们的可见性状态
        return jsonify({
            "keypoints": new_keypoints.tolist(),
            "visibility": visibility.tolist()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 注意：在生产环境中，应使用Gunicorn或uWSGI等WSGI服务器来运行。
    app.run(host='0.0.0.0', port=5000, debug=True)