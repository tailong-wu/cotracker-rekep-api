# cotracker-rekep-api
用于ReKep跟踪的cotracker api部署。

```bash
conda create -n cotracker python=3.10
conda activate cotracker
cd cotracker-rekep-api
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/co-tracker.git
```

首先在终端运行main.py部署api。
使用video_frame_extractor.py对测试视频进行抽帧,生成测试样例。
之后新建一个终端运行test_client.py进行客户端的测试。
注意最少需要15帧，测试样例在frames_test目录下。
测试结果保存在output_frames目录下。

