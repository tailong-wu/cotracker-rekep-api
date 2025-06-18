import torch
import numpy as np
from cotracker.predictor import CoTrackerOnlinePredictor

class CoPointTracker:
    def __init__(self, device=None, window_size=None):
        """
        初始化 CoPointTracker，加载 CoTrackerOnlinePredictor。
        参数:
            device (str or torch.device): 'cuda', 'cpu'，默认自动检测。
            window_size (int): 滑动窗口帧数，None 则自动根据模型 step 推断。
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"CoTracker is running on: {self.device}")

        # 加载模型（官方推荐 hub 方式）
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        self.model = self.model.to(self.device)
        self.window_frames = []
        self.is_first_step = True
        self.window_size = window_size  # 用户可自定义窗口长度
        self.step = self.model.step if hasattr(self.model, "step") else 8  # 默认step

        # 跟踪点缓存，仅第一步用
        self.queries = None

    def reset(self, initial_frame, initial_keypoints, t=0):
        """
        用第一帧和初始追踪点重置跟踪器。
        参数:
            initial_frame (np.ndarray): H x W x 3，uint8/rgb
            initial_keypoints (np.ndarray): N x 2，像素坐标 [x, y]
            t (int): keypoint 所在帧编号，默认0
        """
        self.window_frames = [initial_frame]
        self.is_first_step = True

        # 构造 (1, N, 3): [t, x, y]
        N = initial_keypoints.shape[0]
        points = np.concatenate([np.full((N,1), t), initial_keypoints.astype(np.float32)], axis=1)
        self.queries = torch.from_numpy(points)[None].to(self.device)  # (1, N, 3)

    def append_frame(self, frame):
        """
        添加新帧到窗口，自动维护最大长度
        """
        self.window_frames.append(frame)
        win_len = self.window_size or self.step * 2
        # 保证窗口长度
        if len(self.window_frames) > win_len:
            self.window_frames = self.window_frames[-win_len:]

    def track(self):
        """
        执行一次跟踪（滑动窗口），返回当前窗口所有帧的轨迹结果。
        返回:
            pred_tracks: (1, T, N, 2)
            pred_visibility: (1, T, N)
        """
        if len(self.window_frames) < (self.window_size or self.step * 2):
            raise ValueError("窗口帧数不足，无法进行跟踪。")

        video_chunk = (
            torch.tensor(
                np.stack(self.window_frames), device=self.device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

        kwargs = {
            "video_chunk": video_chunk,
            "is_first_step": self.is_first_step,
        }
        if self.is_first_step:
            kwargs["queries"] = self.queries

        pred_tracks, pred_visibility = self.model(**kwargs)
        self.is_first_step = False  # 后续 step 只传窗口，不传 queries
        return pred_tracks, pred_visibility

    def get_last_frame_results(self, pred_tracks, pred_visibility):
        """
        从模型输出结果提取最新一帧的跟踪点和可见性。
        """
        # pred_tracks: (1, T, N, 2)，取最后一帧
        last_kpts = pred_tracks[0, -1].detach().cpu().numpy()
        last_vis = pred_visibility[0, -1].detach().cpu().numpy()
        return last_kpts, last_vis

    def step_track(self, new_frame):
        """
        常用高阶接口：输入新帧，自动滑窗并返回最新一帧的跟踪点和可见性。
        """
        self.append_frame(new_frame)
        pred_tracks, pred_visibility = self.track()
        return self.get_last_frame_results(pred_tracks, pred_visibility)