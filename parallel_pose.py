import cv2
import math
import numpy as np
import pandas as pd
import mediapipe as mp
import multiprocessing


def process_chunk(video_path, pose_settings, start_frame, end_frame, chunk_id):
    """
    Process frames [start_frame, end_frame) from 'video_path' using MediaPipe Pose.
    Each process runs independently. Returns:
        - chunk_landmarks_data: list of [timestamp, LM1_x, LM1_y, LM1_z, LM1_visibility, ..., LM33_x, LM33_y, ...]
        - chunk_diffs_data: list of [frame_no, diff_x1..diff_x33, diff_y1..diff_y33, slope1..slope33]

    NOTE: We do NOT compute cross-chunk diffs. The first frame in each chunk
          has no 'previous frame' from the prior chunk. If you need continuous diffs,
          you'll need to pass the final XY from chunk i to chunk i+1.
    """
    mp_pose_module = mp.solutions.pose
    pose = mp_pose_module.Pose(**pose_settings)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    chunk_landmarks_data = []
    chunk_diffs_data = []
    prev_xy = None

    frame_count = start_frame  # absolute frame index

    while True:
        if frame_count >= end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Collect absolute landmarks
            frame_landmarks = []
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

            chunk_landmarks_data.append([timestamp] + frame_landmarks)

            # Compute diffs only if we have a previous frame in this chunk
            if prev_xy is not None:
                diff_row = [frame_count - 1]  # store absolute frame_no

                current_xy = np.array([
                    (frame_landmarks[i * 4], frame_landmarks[i * 4 + 1])
                    for i in range(33)
                ])

                diff_x_array = current_xy[:, 0] - prev_xy[:, 0]
                diff_y_array = current_xy[:, 1] - prev_xy[:, 1]
                slope_array = np.zeros(33, dtype=float)

                for i_lm in range(33):
                    dx = diff_x_array[i_lm]
                    dy = diff_y_array[i_lm]
                    if abs(dx) < 1e-3:
                        slope_array[i_lm] = 0
                    else:
                        slope_array[i_lm] = dy / dx

                diff_row.extend(diff_x_array.tolist())
                diff_row.extend(diff_y_array.tolist())
                diff_row.extend(slope_array.tolist())
                chunk_diffs_data.append(diff_row)

                prev_xy = current_xy
            else:
                # This is the first frame in the chunk => no diff from prior chunk
                prev_xy = np.array([
                    (frame_landmarks[i * 4], frame_landmarks[i * 4 + 1])
                    for i in range(33)
                ])
        else:
            # No pose => skip or fill zeros if needed
            pass

    cap.release()
    return chunk_landmarks_data, chunk_diffs_data


def generate_csv_parallel(
        video_path,
        pose_settings,
        output_abs_csv,
        output_diff_csv,
        n_processes=2
):
    """
    Parallel version of CSV generation for MediaPipe Pose:
      1) Determine total_frames and chunk_size.
      2) Spawn n_processes in a Pool, each calls process_chunk(...) for its chunk.
      3) Merge chunk outputs into final CSVs.

    NOTE:
      - We do NOT write a skeleton video in parallel (that would require more advanced merges).
      - If you need it, you can run a separate pass or combine partial videos.

    Returns None, but writes CSV files.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    chunk_size = math.ceil(total_frames / n_processes)

    # Prepare tasks
    tasks = []
    for i in range(n_processes):
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, total_frames)
        if start_frame >= total_frames:
            break
        tasks.append((video_path, pose_settings, start_frame, end_frame, i))

    pool = multiprocessing.Pool(n_processes)
    results = []

    # Launch parallel jobs
    for t in tasks:
        r = pool.apply_async(process_chunk, t)
        results.append(r)

    pool.close()
    pool.join()

    # Combine results
    all_landmarks_data = []
    all_diffs_data = []

    for r in results:
        chunk_landmarks_data, chunk_diffs_data = r.get()
        all_landmarks_data.extend(chunk_landmarks_data)
        all_diffs_data.extend(chunk_diffs_data)

    # Sort by timestamps/frame_no to ensure correct ordering
    all_landmarks_data.sort(key=lambda x: x[0])  # first column is timestamp
    all_diffs_data.sort(key=lambda x: x[0])  # first column is frame_no

    # Prepare column names
    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye',
        'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky',
        'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip',
        'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
        'right_heel', 'left_foot_index', 'right_foot_index'
    ]

    abs_columns = (
            ['timestamp'] +
            [f'{name}_{coord}' for name in landmark_names for coord in ['x', 'y', 'z', 'visibility']]
    )

    diff_columns = (
            ['frame_no'] +
            [f"{name}_diff_x" for name in landmark_names] +
            [f"{name}_diff_y" for name in landmark_names] +
            [f"{name}_slope" for name in landmark_names]
    )

    # Save absolute CSV
    df_abs = pd.DataFrame(all_landmarks_data, columns=abs_columns)
    df_abs.to_csv(output_abs_csv, index=False)

    # Save diffs CSV
    df_diff = pd.DataFrame(all_diffs_data, columns=diff_columns)
    df_diff.to_csv(output_diff_csv, index=False)

    print(f"[Parallel CSV] => {output_abs_csv}, {output_diff_csv} created.")
