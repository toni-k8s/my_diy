import cv2
import os
import time
from datetime import datetime

def record_rtsp_with_opencv(rtsp_url, duration_seconds=None, output_dir=".", prefix="recording_opencv"):
    """
    使用 OpenCV 录制 RTSP 视频流

    Args:
        rtsp_url (str): RTSP 流地址
        duration_seconds (int, optional): 录制持续时间（秒）。如果为 None，则无限录制直到手动中断。
        output_dir (str): 输出文件保存目录，默认为当前目录
        prefix (str): 输出文件名前缀
    """
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"[ERROR] 无法打开 RTSP 流: {rtsp_url}")
        return

    print(f"[INFO] 成功连接到 RTSP 流: {rtsp_url}")

    # 获取视频的基本信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 如果 FPS 无法获取或不合理，设定一个默认值
    if fps <= 0:
        fps = 25
        print(f"[WARNING] 无法获取 FPS，使用默认值: {fps}")

    print(f"[INFO] 视频信息 - FPS: {fps}, Width: {width}, Height: {height}")

    # 定义视频编解码器 (MP4V 是一个常用选择，但可能会产生较大文件)
    # 也可以尝试 'XVID' 等
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') # 另一个选项

    # 生成输出文件名
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp_str}.mp4"
    output_path = os.path.join(output_dir, filename)

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"[ERROR] 无法创建输出视频文件: {output_path}")
        cap.release()
        return

    print(f"[INFO] 开始录制视频到: {output_path}")
    print(f"[INFO] 录制时长: {'无限' if duration_seconds is None else f'{duration_seconds} 秒'}")

    start_time = time.time()
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 无法接收帧 (流可能已断开或结束)")
                break

            # 写入帧到视频文件
            out.write(frame)
            frame_count += 1

            # 检查是否达到指定录制时长
            if duration_seconds is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= duration_seconds:
                    print(f"[INFO] 已达到指定录制时长 ({duration_seconds} 秒)，停止录制。")
                    break

            # # 如果你想实时显示视频（可选）
            # cv2.imshow('Recording RTSP Stream (Press Q to Stop)', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): # 按 'q' 键退出
            #     print("\n[INFO] 用户按下 'q' 键，停止录制。")
            #     break

    except KeyboardInterrupt:
        print("\n[INFO] 用户按下 Ctrl+C，停止录制。")
    finally:
        # 释放资源
        cap.release()
        out.release()
        # cv2.destroyAllWindows() # 如果开启了实时显示窗口才需要取消注释
        print(f"[INFO] 录制结束。总帧数: {frame_count}")
        if frame_count > 0:
            print(f"[INFO] 输出文件: {output_path}")
        else:
            print(f"[WARNING] 录制过程中未捕获到任何帧，可能需要检查 RTSP URL 或网络连接。")


if __name__ == "__main__":
    # --- 配置 ---
    RTSP_URL = "rtsp://admin:gs123456@10.0.1.107:554/cam/realmonitor?channel=1&subtype=0"
    DURATION_SECONDS = None  # 设定录制秒数，例如 60 秒；设为 None 则持续录制直到手动停止
    OUTPUT_DIR = "."         # 输出目录，当前目录为 "."
    FILE_PREFIX = "video_only_opencv" # 输出文件名前缀
    # --- 配置结束 ---

    record_rtsp_with_opencv(RTSP_URL, DURATION_SECONDS, OUTPUT_DIR, FILE_PREFIX)