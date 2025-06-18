import cv2
import argparse
import os

def extract_frames_by_indices(video_path, frame_indices, output_dir):
    """
    按帧编号批量提取视频帧并保存为图像（输出文件名不包含帧编号）

    参数:
    video_path (str): 输入视频文件的路径
    frame_indices (list[int]): 要提取的帧编号列表（从0开始）
    output_dir (str): 输出图像文件的目录
    返回:
    List[str]: 保存的帧图片路径列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    for i, idx in enumerate(frame_indices):
        if idx >= total_frames:
            print(f"警告: 指定帧编号 {idx} 超出视频总帧数 {total_frames}，跳过")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"frame_{i+1:03d}.jpg")
            cv2.imwrite(out_path, frame)
            print(f"已保存第{i+1}帧（编号{idx}）到 {out_path}")
            output_files.append(out_path)
        else:
            print(f"错误: 无法读取第{idx}帧")
    cap.release()
    return output_files

def main():
    parser = argparse.ArgumentParser(description='从视频中按帧编号等间隔提取若干帧图像')
    parser.add_argument('--video', default='./data/apple.mp4', help='输入视频文件路径')
    parser.add_argument('--interval', type=int, default=2, help='帧间隔（每隔多少帧取一帧）')
    parser.add_argument('--max_frames', type=int, default=100, help='最多提取多少帧')
    parser.add_argument('--output_dir', default="frames_test", help='输出图片目录')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 视频文件 {args.video} 不存在")
        return

    # 获取视频总帧数
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 生成要提取的帧编号序列
    frame_indices = list(range(0, total_frames, args.interval))[:args.max_frames]

    output_files = extract_frames_by_indices(args.video, frame_indices, args.output_dir)
    if output_files:
        print(f"成功提取 {len(output_files)} 帧：")
        for f in output_files:
            print(f)
    else:
        print("未能成功提取任何帧。")

if __name__ == "__main__":
    main()