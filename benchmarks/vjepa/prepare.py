#!/usr/bin/env python

import os
import cv2
import numpy as np

def generate_random_video(output_file, width=640, height=480, num_frames=300, fps=30):
    """
    Generates a .mp4 video file with random content.

    :param output_file: Path and name of the output video file
    :param width: Width of the video (in pixels)
    :param height: Height of the video (in pixels)
    :param num_frames: Number of frames in the video
    :param fps: Frames per second (frame rate) of the video
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 encoding
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for _ in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        video_writer.write(frame)

    video_writer.release()


if __name__ == "__main__":
    import sys
    import csv
    import os
    import tqdm
    import multiprocessing

    sys.path.append(os.path.dirname(__file__) + "/jepa/")
    data_directory = os.environ["MILABENCH_DIR_DATA"]
    dest = os.path.join(data_directory, "FakeVideo")
    os.makedirs(dest, exist_ok=True)

    csv_file = os.path.join(dest, "video_metainfo.csv")
    
    num_videos = 1000  # Change this to generate more or fewer videos
    num_frames = 300

    def gen_video(i):
        output_file = os.path.join(dest, f"{i + 1}.mp4")
        if not os.path.exists(output_file):
            generate_random_video(output_file=output_file, width=640, height=480, num_frames=num_frames, fps=30)
        
    n_worker = min(multiprocessing.cpu_count(), 16)

    with multiprocessing.Pool(n_worker) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(gen_video, range(num_videos)), total=num_videos):
            pass

    with open(csv_file, mode='w', newline='') as file:
        # CSV separated by space genius
        writer = csv.writer(file, delimiter=" ")
        for file in tqdm.tqdm(os.listdir(dest)):
            if file.endswith(".mp4"):
                writer.writerow([os.path.join(dest, file), 0])

    print(f"Generated {num_videos} videos and created {csv_file}")

    # If there is nothing to download or generate, just delete this file.
