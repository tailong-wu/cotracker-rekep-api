import torch
# Download the video
url = 'https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4'

import imageio.v3 as iio
frames = iio.imread(url, plugin="FFMPEG")  # plugin="pyav"

device = 'cuda'
grid_size = 10
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

# Run Offline CoTracker:
# cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
# pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1

# Run Online CoTracker, the same model with a different API:
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

# Initialize online processing
cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)  

# Process the video
for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
    pred_tracks, pred_visibility = cotracker(
        video_chunk=video[:, ind : ind + cotracker.step * 2]
    )  # B T N 2,  B T N 1