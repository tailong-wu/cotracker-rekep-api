# cotracker-rekep-api
用于ReKep跟踪的cotracker api部署。

'''bash
conda create -n cotracker python=3.10
conda activate cotracker
cd cotracker-rekep-api
pip install -r requiremnets.txt
'''

首先在终端运行main.py部署api。
使用video_frame_extractor.py对测试视频进行抽帧。
之后新建一个终端运行test_client.py进行客户端的测试。

