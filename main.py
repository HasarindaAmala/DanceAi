import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
import matplotlib.pyplot as plt
import multiprocessing  # only needed if you call your parallel generator here
from parallel_pose import generate_csv_parallel
from vocal import VocalSync

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_similarity_diff_csv(ref_csv, test_csv):
    """
    Compare two 'diff CSV' files from 33 landmarks.
    Each file has columns:
      frame_no,
      [33 columns of _diff_x],
      [33 columns of _diff_y],
      [33 columns of _slope].

    Produces two arrays:
      dist_diff_array (n_frames,) => L2 norm for position diffs
      angle_diff_array (n_frames,) => L2 norm for (arctan slopes) diffs
    """
    df_ref = pd.read_csv(ref_csv)
    df_test = pd.read_csv(test_csv)

    min_len = min(len(df_ref), len(df_test))
    df_ref = df_ref.iloc[:min_len]
    df_test = df_test.iloc[:min_len]

    # Indices for each body part (based on 33 pose landmarks)
    arm_indices = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
    leg_indices = [23, 24, 25, 26, 27, 28]  # hips, knees, ankles
    mid_indices = [11, 12, 23, 24]  # shoulders & hips

    dx_start = 1
    dx_end = dx_start + 33  # 1..33
    dy_start = dx_end
    dy_end = dy_start + 33  # 34..66
    slope_start = dy_end  # 67
    slope_end = slope_start + 33  # 100

    # Initialize arrays for each category

    dist_diff_array = np.zeros(min_len, dtype=float)
    angle_diff_array = np.zeros(min_len, dtype=float)

    dist_diff_arms = np.zeros(min_len, dtype=float)
    angle_diff_arms = np.zeros(min_len, dtype=float)

    dist_diff_legs = np.zeros(min_len, dtype=float)
    angle_diff_legs = np.zeros(min_len, dtype=float)

    dist_diff_mid = np.zeros(min_len, dtype=float)
    angle_diff_mid = np.zeros(min_len, dtype=float)

    for i in range(min_len):
        row_ref = df_ref.iloc[i].values
        row_test = df_test.iloc[i].values

        ref_dx = row_ref[dx_start: dx_end]
        ref_dy = row_ref[dy_start: dy_end]
        ref_slope = row_ref[slope_start: slope_end]

        test_dx = row_test[dx_start: dx_end]
        test_dy = row_test[dy_start: dy_end]
        test_slope = row_test[slope_start: slope_end]

        # 1) Distance-based difference
        diff_dx = ref_dx - test_dx
        diff_dy = ref_dy - test_dy
        diff_dist = np.concatenate([diff_dx, diff_dy])
        dist_diff = np.linalg.norm(diff_dist)
        dist_diff_array[i] = dist_diff

        # 2) Angle-based difference
        angle_ref = np.arctan(ref_slope)
        angle_test = np.arctan(test_slope)
        diff_angle = angle_ref - angle_test
        angle_diff = np.linalg.norm(diff_angle)
        angle_diff_array[i] = angle_diff

        # Extract values for each part
        for part, dist_diff_array, angle_diff_array, indices in [
            ("arms", dist_diff_arms, angle_diff_arms, arm_indices),
            ("legs", dist_diff_legs, angle_diff_legs, leg_indices),
            ("mid", dist_diff_mid, angle_diff_mid, mid_indices)
        ]:
            ref_dx = row_ref[[idx + 1 for idx in indices]]  # X differences
            ref_dy = row_ref[[idx + 34 for idx in indices]]  # Y differences
            ref_slope = row_ref[[idx + 67 for idx in indices]]  # Slopes

            test_dx = row_test[[idx + 1 for idx in indices]]
            test_dy = row_test[[idx + 34 for idx in indices]]
            test_slope = row_test[[idx + 67 for idx in indices]]

            # Distance-based difference
            diff_dx = ref_dx - test_dx
            diff_dy = ref_dy - test_dy
            diff_dist = np.concatenate([diff_dx, diff_dy])
            dist_diff = np.linalg.norm(diff_dist)
            dist_diff_array[i] = dist_diff

            # Angle-based difference
            angle_ref = np.arctan(ref_slope)
            angle_test = np.arctan(test_slope)
            diff_angle = angle_ref - angle_test
            angle_diff = np.linalg.norm(diff_angle)
            angle_diff_array[i] = angle_diff

    return (dist_diff_array, angle_diff_array, dist_diff_arms, angle_diff_arms,
            dist_diff_legs, angle_diff_legs,
            dist_diff_mid, angle_diff_mid)


def smooth_signal(signal, window_size=5):
    """Simple moving average smoothing for angle array."""
    if len(signal) < window_size:
        return signal

    smoothed = np.convolve(signal, np.ones(window_size) / window_size, mode='valid')
    # Pad so output length == input length
    return np.concatenate((signal[:window_size - 1], smoothed))


def genarate_marks(similarities, angle):
    """
    Convert distance/angle arrays to 0-100 scale:
      0 => worst difference, 100 => best (no difference).
    """
    max_score_dis = max(similarities) if len(similarities) else 1
    max_score_angle = max(angle) if len(angle) else 1

    if max_score_dis == 0:
        max_score_dis = 1
    if max_score_angle == 0:
        max_score_angle = 1

    marks_dis = [100 - (score / max_score_dis * 100) for score in similarities]
    marks_angle = [100 - (score / max_score_angle * 100) for score in angle]
    return marks_dis, marks_angle


# Define function to draw glowing skeleton lines
def draw_glowing_skeleton(overlay, start, end, color, thickness=4):
    """
    Draws a glowing line between two points using an overlay.
    """
    cv2.line(overlay, start, end, color, thickness, cv2.LINE_AA)


# Define function to generate slow-motion glowing overlay video
def generate_glowing_overlay_video_slow(test_video_path, reference_video_path,
                                        reference_landmarks_csv, test_landmarks_csv,
                                        output_video_path, speed_factor=0.5):
    """
    Generates an overlay video with a glowing skeleton, motion trails, reduced opacity,
    and slows down the video by the given speed_factor (default = 0.5x).
    """
    cap = cv2.VideoCapture(test_video_path)
    cap_ref = cv2.VideoCapture(reference_video_path)
    landmarks_df = pd.read_csv(reference_landmarks_csv)
    landmarks_test = pd.read_csv(test_landmarks_csv)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(landmarks_df), len(landmarks_test))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    mp_pose = mp.solutions.pose
    pose_connections = list(mp_pose.POSE_CONNECTIONS)

    # Ignore face-related landmarks (only consider body landmarks)
    face_landmark_indices = list(range(0, 10))  # Mediapipe face indices
    pose_connections = [conn for conn in pose_connections if
                        conn[0] not in face_landmark_indices and conn[1] not in face_landmark_indices]

    # Motion trail buffer
    motion_trail = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    for frame_count in tqdm(range(total_frames), desc=f"Generating glowing overlay -> {output_video_path}"):
        ret, frame = cap.read()
        if not ret or frame_count >= len(landmarks_df):
            break

        ref_row = landmarks_df.iloc[frame_count]
        test_row = landmarks_test.iloc[frame_count]

        # Body height normalization
        ref_hip = (ref_row['right_hip_y'] + ref_row['left_hip_y']) / 2
        ref_shoulder = (ref_row['right_shoulder_y'] + ref_row['left_shoulder_y']) / 2
        ref_height = abs(ref_shoulder - ref_hip)

        test_hip = (test_row['right_hip_y'] + test_row['left_hip_y']) / 2
        test_shoulder = (test_row['right_shoulder_y'] + test_row['left_shoulder_y']) / 2
        test_height = abs(test_shoulder - test_hip)

        if ref_height < 1e-6:
            ref_height = 1e-6

        scale_factor = test_height / ref_height
        scale_factor = max(1.35, min(scale_factor, 1.2))

        # Body centers
        ref_center_x = (ref_row['right_shoulder_x'] + ref_row['left_shoulder_x'] +
                        ref_row['right_hip_x'] + ref_row['left_hip_x']) / 4
        ref_center_y = (ref_row['right_shoulder_y'] + ref_row['left_shoulder_y'] +
                        ref_row['right_hip_y'] + ref_row['left_hip_y']) / 4

        test_center_x = (test_row['right_shoulder_x'] + test_row['left_shoulder_x'] +
                         test_row['right_hip_x'] + test_row['left_hip_x']) / 4
        test_center_y = (test_row['right_shoulder_y'] + test_row['left_shoulder_y'] +
                         test_row['right_hip_y'] + test_row['left_hip_y']) / 4

        x_offset = (test_center_x * frame_width) - (ref_center_x * frame_width_ref * scale_factor)
        y_offset = (test_center_y * frame_height) - (ref_center_y * frame_height_ref * scale_factor)

        ref_landmarks = ref_row[1:].values.reshape(-1, 4)
        test_landmarks = test_row[1:].values.reshape(-1, 4)

        overlay = np.zeros_like(frame, dtype=np.uint8)

        for connection in pose_connections:
            start_idx, end_idx = connection
            rx1, ry1, _, rvis1 = ref_landmarks[start_idx]
            rx2, ry2, _, rvis2 = ref_landmarks[end_idx]

            tx1, ty1, _, tvis1 = test_landmarks[start_idx]
            tx2, ty2, _, tvis2 = test_landmarks[end_idx]

            if rvis1 > 0.5 and rvis2 > 0.5 and tvis1 > 0.5 and tvis2 > 0.5:
                px1_ref = int(rx1 * frame_width_ref * scale_factor + x_offset)
                py1_ref = int(ry1 * frame_height_ref * scale_factor + y_offset)
                px2_ref = int(rx2 * frame_width_ref * scale_factor + x_offset)
                py2_ref = int(ry2 * frame_height_ref * scale_factor + y_offset)

                mid_ref_x = 0.5 * (px1_ref + px2_ref)
                mid_ref_y = 0.5 * (py1_ref + py2_ref)

                mid_test_x = 0.5 * ((tx1 * frame_width)) + 0.5 * ((tx2 * frame_width))
                mid_test_y = 0.5 * ((ty1 * frame_height)) + 0.5 * ((ty2 * frame_height))

                dx = mid_test_x - mid_ref_x
                dy = mid_test_y - mid_ref_y
                dist = np.sqrt(dx ** 2 + dy ** 2)

                if dist < 50:
                    line_color = (0, 255, 0)  # Green (Accurate)
                elif dist > 90:
                    line_color = (0, 0, 255)  # Red (Large Deviation)
                else:
                    line_color = (0, 165, 255)  # Orange (Moderate Deviation)

                thickness = max(2, int(6 - (dist / 50)))  # Thicker for accurate moves
                draw_glowing_skeleton(overlay, (px1_ref, py1_ref), (px2_ref, py2_ref), line_color, thickness)

        # Apply blur once for the entire skeleton glow
        glow_overlay = cv2.GaussianBlur(overlay, (13, 13), 0)

        # Motion Trail Effect - Keep a faded previous overlay
        motion_trail = cv2.addWeighted(motion_trail, 0.1, glow_overlay, 0.8, 0)

        # Combine layers
        final_overlay = cv2.addWeighted(frame, 0.8, motion_trail, 0.7, 0)

        # **Slow Motion Effect: Duplicate Frames**
        for _ in range(int(1 / speed_factor)):  # 2x frame duplication for 0.5x speed
            out.write(final_overlay)

    cap.release()
    cap_ref.release()
    out.release()
    print(f"Glowing overlay video with motion trails (0.5x speed) saved to {output_video_path}")


def generate_overlay_video(test_video_path, reference_video_path,
                           reference_landmarks_csv, test_landmarks_csv,
                           output_video_path):
    """
    Overlays the reference skeleton (colored lines, joint markers)
    onto the test video, plus draws red circle for reference center
    and blue circle for test center. Annotates segments with error
    magnitude if the error is high so that the user can visually identify
    mistakes.
    """
    cap = cv2.VideoCapture(test_video_path)
    cap_ref = cv2.VideoCapture(reference_video_path)
    landmarks_df = pd.read_csv(reference_landmarks_csv)
    landmarks_test = pd.read_csv(test_landmarks_csv)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_landmark_frames = min(len(landmarks_df), len(landmarks_test))
    total_frames = min(total_frames, total_landmark_frames)

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    mp_pose_module = mp.solutions.pose
    pose_connections = mp_pose_module.POSE_CONNECTIONS

    for frame_count in tqdm(range(total_frames), desc=f"Generating overlay -> {output_video_path}"):
        ret, frame = cap.read()
        if not ret or frame_count >= len(landmarks_df):
            break

        ref_row = landmarks_df.iloc[frame_count]
        test_row = landmarks_test.iloc[frame_count]

        # Full body height for dynamic scale.
        ref_hip = (ref_row['right_hip_y'] + ref_row['left_hip_y']) / 2
        ref_shoulder = (ref_row['right_shoulder_y'] + ref_row['left_shoulder_y']) / 2
        ref_height = abs(ref_shoulder - ref_hip)

        test_hip = (test_row['right_hip_y'] + test_row['left_hip_y']) / 2
        test_shoulder = (test_row['right_shoulder_y'] + test_row['left_shoulder_y']) / 2
        test_height = abs(test_shoulder - test_hip)

        if ref_height < 1e-6:
            ref_height = 1e-6

        # Compute scale factor; ensuring it's between 1.2 and 1.3.
        scale_factor = test_height / ref_height
        scale_factor = max(1.3, min(scale_factor, 1.4))

        # Compute body centers (normalized coordinates).
        ref_center_x = (ref_row['right_shoulder_x'] + ref_row['left_shoulder_x'] +
                        ref_row['right_hip_x'] + ref_row['left_hip_x']) / 4
        ref_center_y = (ref_row['right_shoulder_y'] + ref_row['left_shoulder_y'] +
                        ref_row['right_hip_y'] + ref_row['left_hip_y']) / 4

        test_center_x = (test_row['right_shoulder_x'] + test_row['left_shoulder_x'] +
                         test_row['right_hip_x'] + test_row['left_hip_x']) / 4
        test_center_y = (test_row['right_shoulder_y'] + test_row['left_shoulder_y'] +
                         test_row['right_hip_y'] + test_row['left_hip_y']) / 4

        # Compute offsets so the reference skeleton aligns onto the test video.
        x_offset = (test_center_x * frame_width) - (ref_center_x * frame_width_ref * scale_factor)
        y_offset = (test_center_y * frame_height) - (ref_center_y * frame_height_ref * scale_factor)

        # Reshape landmarks (assuming columns 1 onward are the coordinates).
        ref_landmarks = ref_row[1:].values.reshape(-1, 4)
        test_landmarks = test_row[1:].values.reshape(-1, 4)

        # Draw reference skeleton segments.
        for connection in pose_connections:
            start_idx, end_idx = connection
            # Get reference landmarks.
            rx1, ry1, _, rvis1 = ref_landmarks[start_idx]
            rx2, ry2, _, rvis2 = ref_landmarks[end_idx]
            # Get test landmarks (for error measurement).
            tx1, ty1, _, tvis1 = test_landmarks[start_idx]
            tx2, ty2, _, tvis2 = test_landmarks[end_idx]

            if rvis1 > 0.5 and rvis2 > 0.5 and tvis1 > 0.5 and tvis2 > 0.5:
                # Convert normalized reference coordinates to pixel coordinates.
                px1_ref = int(rx1 * frame_width_ref * scale_factor + x_offset)
                py1_ref = int(ry1 * frame_height_ref * scale_factor + y_offset)
                px2_ref = int(rx2 * frame_width_ref * scale_factor + x_offset)
                py2_ref = int(ry2 * frame_height_ref * scale_factor + y_offset)

                # Compute midpoints for error calculation.
                mid_ref_x = 0.5 * (px1_ref + px2_ref)
                mid_ref_y = 0.5 * (py1_ref + py2_ref)
                mid_test_x = 0.5 * (tx1 * frame_width + tx2 * frame_width)
                mid_test_y = 0.5 * (ty1 * frame_height + ty2 * frame_height)

                dx = mid_test_x - mid_ref_x
                dy = mid_test_y - mid_ref_y
                dist = np.sqrt(dx ** 2 + dy ** 2)

                # Choose line color based on error distance.
                if dist < 50:
                    line_color = (0, 255, 0)  # Green: good alignment.
                elif dist > 90:
                    line_color = (0, 0, 255)  # Red: large error.
                else:
                    line_color = (0, 165, 255)  # Orange: moderate error.

                # Draw the skeleton segment with thicker line.
                cv2.line(frame, (px1_ref, py1_ref), (px2_ref, py2_ref), line_color, thickness=12)

        # Draw circles at each reference joint.
        for i in range(len(ref_landmarks)):
            rx, ry, _, rvis = ref_landmarks[i]
            if rvis > 0.5:
                px = int(rx * frame_width_ref * scale_factor + x_offset)
                py = int(ry * frame_height_ref * scale_factor + y_offset)
                cv2.circle(frame, (px, py), radius=6, color=(255, 255, 255), thickness=-1)

        # Draw center circles.
        ref_center_px = int(ref_center_x * frame_width_ref * scale_factor + x_offset)
        ref_center_py = int(ref_center_y * frame_height_ref * scale_factor + y_offset)
        test_center_px = int(test_center_x * frame_width)
        test_center_py = int(test_center_y * frame_height)

        cv2.circle(frame, (ref_center_px, ref_center_py), radius=6, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (test_center_px, test_center_py), radius=6, color=(255, 0, 0), thickness=-1)

        out.write(frame)

    cap.release()
    cap_ref.release()
    out.release()
    print(f"Optimized overlay video saved to {output_video_path}")


def plot_distance_vs_frame(distance_array, title="Distance Differences Over Frames"):
    """
    Plots the distance differences (L2 norm) across frames.

    :param distance_array: 1D array-like of distance differences
    :param title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    frames = range(len(distance_array))

    # Plot the distance values
    plt.plot(frames, distance_array, color='blue', label='Distance Difference')

    plt.xlabel('Frame Number')
    plt.ylabel('Distance Difference (L2 norm)')
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.show()
    # If you want to save to a file, uncomment:
    # plt.savefig('distance_diff_plot.png', dpi=150)
    # plt.close()


def generate_side_by_side_heatmap_with_score(
        ref_video, test_video,
        ref_csv, test_csv,
        output_video,

):
    """
    Creates a side-by-side video:
      - Left: reference video frames
      - Right: test video frames with a 'heatmap circle' on each landmark
      - Overall marks displayed at the top
    """
    cap_ref = cv2.VideoCapture(ref_video)
    cap_test = cv2.VideoCapture(test_video)

    df_ref = pd.read_csv(ref_csv)
    df_test = pd.read_csv(test_csv)
    min_frames = min(int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT)),
                     len(df_ref), len(df_test))

    fps = int(cap_test.get(cv2.CAP_PROP_FPS))
    fps_slow = max(1, int(fps * 0.75))
    w_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))

    w_test = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_test = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))

    final_width = w_ref + w_test
    final_height = max(h_ref, h_test)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps_slow, (final_width, final_height))

    for frame_idx in tqdm(range(min_frames), desc=f"Generating side-by-side heatmap -> {output_video}"):
        ret_ref, frame_ref = cap_ref.read()
        ret_test, frame_test = cap_test.read()
        if not (ret_ref and ret_test):
            break

        if h_ref != h_test:
            frame_ref = cv2.resize(frame_ref, (w_ref, final_height))
            frame_test = cv2.resize(frame_test, (w_test, final_height))

        combined_frame = np.zeros((final_height, final_width, 3), dtype=np.uint8)
        combined_frame[:, :w_ref] = frame_ref
        combined_frame[:, w_ref:] = frame_test

        # Add final score on top of the video

        out.write(combined_frame)

    cap_ref.release()
    cap_test.release()
    out.release()
    print(f"Side-by-side heatmap video with score saved -> {output_video}")


def generate_2x2_matrix_video(
        ref_video, test_video,
        ref_csv, test_csv,
        output_video,
        score_arms, score_legs, score_mid,
        max_error_threshold=0.4
):
    """
    Creates a 2x2 matrix video:
      - Top-left: Reference video
      - Top-right: Test video with heatmap (Arms)
      - Bottom-left: Test video with heatmap (Legs)
      - Bottom-right: Test video with heatmap (Mid-Body)
      - Each video section has its corresponding score displayed
    """
    cap_ref = cv2.VideoCapture(ref_video)
    cap_test = cv2.VideoCapture(test_video)

    df_ref = pd.read_csv(ref_csv)
    df_test = pd.read_csv(test_csv)
    min_frames = min(int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT)),
                     len(df_ref), len(df_test))

    fps = int(cap_ref.get(cv2.CAP_PROP_FPS))
    width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))

    final_width = width * 2  # 2 columns
    final_height = height * 2  # 2 rows

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (final_width, final_height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 3
    text_color = (255, 255, 255)

    for frame_idx in tqdm(range(min_frames), desc=f"Generating 2x2 Matrix Video -> {output_video}"):
        ret_ref, frame_ref = cap_ref.read()
        ret_test, frame_test = cap_test.read()
        if not (ret_ref and ret_test):
            break

        frame_ref = cv2.resize(frame_ref, (width, height))
        frame_test = cv2.resize(frame_test, (width, height))

        frame_arms = frame_test.copy()
        frame_legs = frame_test.copy()
        frame_mid = frame_test.copy()

        row_ref = df_ref.iloc[frame_idx].values
        row_test = df_test.iloc[frame_idx].values

        arm_indices = [11, 12, 13, 14, 15, 16]
        leg_indices = [23, 24, 25, 26, 27, 28]
        mid_indices = [11, 12, 23, 24]

        for part, frame, indices in [
            ("arms", frame_arms, arm_indices),
            ("legs", frame_legs, leg_indices),
            ("mid", frame_mid, mid_indices)
        ]:
            for lm_id in indices:
                ref_x = row_ref[1 + 4 * lm_id]
                ref_y = row_ref[2 + 4 * lm_id]
                ref_vis = row_ref[4 + 4 * lm_id]

                test_x = row_test[1 + 4 * lm_id]
                test_y = row_test[2 + 4 * lm_id]
                test_vis = row_test[4 + 4 * lm_id]

                if ref_vis < 0.5 or test_vis < 0.5:
                    continue

                px_ref_x = int(ref_x * width)
                px_ref_y = int(ref_y * height)
                px_test_x = int(test_x * width)
                px_test_y = int(test_y * height)

                dx = px_test_x - px_ref_x
                dy = px_test_y - px_ref_y
                dist = np.sqrt(dx * dx + dy * dy)

                normalized_dist = dist / float(width)

                target_x = px_test_x
                target_y = px_test_y

                if normalized_dist < max_error_threshold / 2:
                    circle_color = (0, 255, 0)
                elif normalized_dist > max_error_threshold:
                    circle_color = (0, 0, 255)
                else:
                    circle_color = (0, 165, 255)

                cv2.circle(frame, (target_x, target_y), 10, circle_color, -1)

        top_row = np.hstack((frame_ref, frame_arms))
        bottom_row = np.hstack((frame_legs, frame_mid))
        final_frame = np.vstack((top_row, bottom_row))

        # Add Scores on Each Section
        scores = [
            ("Reference", frame_ref, (50, 50)),
            (f"Arms Score: {score_arms:.2f}", frame_arms, (width + 50, 50)),
            (f"Legs Score: {score_legs:.2f}", frame_legs, (50, height + 50)),
            (f"Mid-Body Score: {score_mid:.2f}", frame_mid, (width + 50, height + 50)),
        ]

        for text, _, position in scores:
            cv2.putText(final_frame, text, position, font, font_scale, text_color, font_thickness)

        out.write(final_frame)

    cap_ref.release()
    cap_test.release()
    out.release()
    print(f"2x2 matrix video saved -> {output_video}")


def scoreScreen(output_video, ref, score_full, score_arm, score_legs, score_mid):
    """
    Generates a 10-second black screen video displaying scores in the center.
    """
    cap = cv2.VideoCapture(ref)

    # Extract frame rate and resolution from the reference video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        print("Error: Could not retrieve valid FPS or video dimensions.")
        cap.release()
        return

    cap.release()  # Close reference video

    # Video settings
    duration = 10  # seconds
    total_frames = fps * duration

    # Use a more compatible codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'XVID' is widely supported
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 5
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black background

    # Score Texts
    scores = [
        (f"Final Score: {score_full:.2f}", (width // 4, height // 4)),
        (f"Arms Score: {score_arm:.2f}", (width // 4, height // 2 - 50)),
        (f"Legs Score: {score_legs:.2f}", (width // 4, height // 2 + 50)),
        (f"Mid-Body Score: {score_mid:.2f}", (width // 4, height // 2 + 150)),
    ]

    for _ in range(total_frames):
        # Create a black frame with explicit RGB color space
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Overlay text on the black frame
        for text, position in scores:
            cv2.putText(black_frame, text, position, font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

        out.write(black_frame)

    out.release()
    print(f"Score screen video saved -> {output_video}")


def generate_live_score_video(
        overlay_video, output_video,
        ref_csv, test_csv,
        fps
):
    """
    Generates a video that overlays a dynamically calculated live score
    at 0.5-second intervals while playing the given overlay video.
    The score for each interval is calculated from only the frames within that 0.5s block.
    """
    cap = cv2.VideoCapture(overlay_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Use the video's own FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, video_fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black background for score text

    # 0.5-second interval in number of frames
    update_interval = int(video_fps / 2)

    # Read landmark difference CSVs
    df_ref = pd.read_csv(ref_csv)
    df_test = pd.read_csv(test_csv)
    min_len = min(len(df_ref), len(df_test))
    df_ref = df_ref.iloc[:min_len]
    df_test = df_test.iloc[:min_len]

    # Define indices for each body part (based on 33 pose landmarks)
    arm_indices = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
    leg_indices = [23, 24, 25, 26, 27, 28]  # hips, knees, ankles
    mid_indices = [11, 12, 23, 24]  # shoulders & hips

    # Initialize score variables (they will update each block)
    live_score_full = live_score_arm = live_score_legs = live_score_mid = 0

    overall_score_distance_full = overall_score_distance_arm = overall_score_distance_legs = overall_score_distance_mid = 0

    # Process each frame.
    # Helper function to extract X and Y differences from a block
    def extract_landmark_diff(df, indices, start_frame, end_frame):
        cols_x = [1 + idx for idx in indices]  # X-difference columns
        cols_y = [34 + idx for idx in indices]  # Y-difference columns
        return df.iloc[start_frame:end_frame, cols_x + cols_y].values

    for frame_idx in tqdm(range(total_frames), desc=f"Generating overlay -> {output_video}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Update scores every 0.5 seconds (every update_interval frames).
        if frame_idx % update_interval == 0 and frame_idx != 0:
            start_frame = frame_idx
            # Use exactly 0.5s worth of frames (or as many remain)
            end_frame = min(frame_idx + update_interval, min_len)

            # For full-body, assume the CSV columns (after the frame number) contain all differences
            ref_full = df_ref.iloc[start_frame:end_frame, 1:].values
            test_full = df_test.iloc[start_frame:end_frame, 1:].values

            # Extract differences for arms, legs, and mid
            ref_arms = extract_landmark_diff(df_ref, arm_indices, start_frame, end_frame)
            test_arms = extract_landmark_diff(df_test, arm_indices, start_frame, end_frame)
            ref_legs = extract_landmark_diff(df_ref, leg_indices, start_frame, end_frame)
            test_legs = extract_landmark_diff(df_test, leg_indices, start_frame, end_frame)
            ref_mid = extract_landmark_diff(df_ref, mid_indices, start_frame, end_frame)
            test_mid = extract_landmark_diff(df_test, mid_indices, start_frame, end_frame)

            # Compute Euclidean differences for each frame in the block
            dist_full = np.linalg.norm(ref_full - test_full, axis=1)
            dist_arms = np.linalg.norm(ref_arms - test_arms, axis=1)
            dist_legs = np.linalg.norm(ref_legs - test_legs, axis=1)
            dist_mid = np.linalg.norm(ref_mid - test_mid, axis=1)

            # Compute mean differences over the interval
            mean_diff_full = np.mean(dist_full)
            mean_diff_arms = np.mean(dist_arms)
            mean_diff_legs = np.mean(dist_legs)
            mean_diff_mid = np.mean(dist_mid)

            # --- New Normalization using fixed thresholds ---
            # These constants can be tuned based on your typical error ranges.


            live_score_full = max(0, 100 - (mean_diff_full * 100 / max(1, np.max(dist_full))))
            live_score_arm = max(0, 100 - (mean_diff_arms * 100 / max(1, np.max(dist_arms))))
            live_score_legs = max(0, 100 - (mean_diff_legs * 100 / max(1, np.max(dist_legs))))
            live_score_mid = max(0, 100 - (mean_diff_mid * 100 / max(1, np.max(dist_mid))))


        # Create a black rectangle for score display.
        cv2.rectangle(frame, (10, 10), (370, 220), bg_color, -1)

        # Overlay the current scores on the frame.
        cv2.putText(frame, f"Full: {live_score_full:.2f}", (20, 40), font, font_scale, text_color,
                    font_thickness)
        cv2.putText(frame, f"Arms: {live_score_arm:.2f}", (20, 90), font, font_scale, text_color,
                    font_thickness)
        cv2.putText(frame, f"Legs: {live_score_legs:.2f}", (20, 140), font, font_scale, text_color,
                    font_thickness)
        cv2.putText(frame, f"Mid:  {live_score_mid:.2f}", (20, 190), font, font_scale, text_color,
                    font_thickness)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Live score video saved -> {output_video}")


# def overlay_cartoon_character(
#         ref_video, test_video, output_video, landmarks_csv, character_parts, fps
# ):
#     """
#     Overlays a cartoon character onto the test video, matching the reference video movements.
#     """
#     cap_test = cv2.VideoCapture(test_video)
#     cap_ref = cv2.VideoCapture(ref_video)
#     width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
#
#     # Load character parts (each must be a transparent PNG)
#     parts = {part: cv2.imread(character_parts[part], cv2.IMREAD_UNCHANGED) for part in character_parts}
#
#     # Read pose landmarks
#     df_landmarks = pd.read_csv(landmarks_csv)
#     min_frames = min(len(df_landmarks), total_frames)
#
#     def overlay_image(background, overlay, x, y, scale=1.0, rotation=0):
#         """Overlays a transparent image onto the background at a given position, scale, and rotation."""
#         if overlay is None:
#             return background
#
#         # Resize
#         overlay_resized = cv2.resize(overlay, None, fx=scale, fy=scale)
#         h, w, _ = overlay_resized.shape
#
#         # Rotate
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, rotation, 1.0)
#         overlay_rotated = cv2.warpAffine(overlay_resized, M, (w, h))
#
#         # Compute position
#         y1, y2 = max(0, y - h // 2), min(height, y + h // 2)
#         x1, x2 = max(0, x - w // 2), min(width, x + w // 2)
#
#         # Extract alpha channel
#         alpha_s = overlay_rotated[:, :, 3] / 255.0
#         alpha_l = 1.0 - alpha_s
#
#         # Overlay character part
#         for c in range(3):
#             background[y1:y2, x1:x2, c] = (
#                     alpha_s * overlay_rotated[:, :, c] + alpha_l * background[y1:y2, x1:x2, c]
#             )
#
#         return background
#
#     for frame_idx in range(min_frames):
#         ret_test, frame = cap_test.read()
#         ret_ref, _ = cap_ref.read()
#         if not ret_test:
#             break
#
#         # Extract relevant landmarks
#         row = df_landmarks.iloc[frame_idx]
#
#         # Get body part positions
#         nose_x, nose_y = int(row["nose_x"] * width), int(row["nose_y"] * height)
#         hip_x, hip_y = int((row["right_hip_x"] + row["left_hip_x"]) / 2 * width), int(
#             (row["right_hip_y"] + row["left_hip_y"]) / 2 * height)
#         shoulder_x, shoulder_y = int((row["right_shoulder_x"] + row["left_shoulder_x"]) / 2 * width), int(
#             (row["right_shoulder_y"] + row["left_shoulder_y"]) / 2 * height)
#
#         # Compute rotation angles
#         arm_angle = np.degrees(
#             np.arctan2(row["right_elbow_y"] - row["right_shoulder_y"], row["right_elbow_x"] - row["right_shoulder_x"]))
#         leg_angle = np.degrees(
#             np.arctan2(row["right_knee_y"] - row["right_hip_y"], row["right_knee_x"] - row["right_hip_x"]))
#
#         # Overlay character parts
#         frame = overlay_image(frame, parts["head"], nose_x, nose_y, scale=0.6)
#         frame = overlay_image(frame, parts["torso"], hip_x, hip_y, scale=1.0)
#         frame = overlay_image(frame, parts["left_arm"], shoulder_x - 40, shoulder_y, scale=0.5, rotation=arm_angle)
#         frame = overlay_image(frame, parts["right_arm"], shoulder_x + 40, shoulder_y, scale=0.5, rotation=-arm_angle)
#         frame = overlay_image(frame, parts["left_leg"], hip_x - 30, hip_y + 50, scale=0.5, rotation=leg_angle)
#         frame = overlay_image(frame, parts["right_leg"], hip_x + 30, hip_y + 50, scale=0.5, rotation=-leg_angle)
#
#         out.write(frame)
#
#     cap_test.release()
#     cap_ref.release()
#     out.release()
#     print(f"Cartoon character animation saved -> {output_video}")
#


if __name__ == "__main__":
    # 1) Sync videos by audio
    # vocal_obj = VocalSync("reference.mp4", "test.mp4")
    # ref_sync, test_sync = vocal_obj.start_vocalSync()
    #
    # # 2) Parallel Pose settings
    # pose_settings = dict(
    #     static_image_mode=False,
    #     model_complexity=1,  # lower => faster
    #     smooth_landmarks=True,
    #     min_detection_confidence=0.5,
    #     min_tracking_confidence=0.5
    # )
    #
    # # 3) Generate CSV in parallel for reference
    # generate_csv_parallel(
    #     video_path="reference_synced.mp4",
    #     pose_settings=pose_settings,
    #     output_abs_csv="landmarks_with_timestamps_ref.csv",
    #     output_diff_csv="diff_landmarks_ref.csv",
    #     n_processes=4  # Adjust based on number of CPU cores
    # )
    #
    # # 4) Generate CSV in parallel for test
    # generate_csv_parallel(
    #     video_path="test_synced.mp4",
    #     pose_settings=pose_settings,
    #     output_abs_csv="landmarks_with_timestamps_test.csv",
    #     output_diff_csv="diff_landmarks_test.csv",
    #     n_processes=4
    # )

    # # 5) Compare CSVs => produce distance & angle arrays
    # distance_array, angle_array,distance_array_arms, angle_array_arms,distance_array_legs, angle_array_legs,distance_array_mid, angle_array_mid = calculate_similarity_diff_csv(
    #     "diff_landmarks_ref.csv",
    #     "diff_landmarks_test.csv"
    # )
    #
    # # 6) Smooth angle array (optional)
    # smooth_angle = smooth_signal(angle_array)
    # smooth_angle_arms = smooth_signal(angle_array_arms)
    # smooth_angle_legs = smooth_signal(angle_array_legs)
    # smooth_angle_mid = smooth_signal(angle_array_mid)
    #
    # # 7) Generate marks for full body
    # score_distance, score_angle = genarate_marks(distance_array, smooth_angle)
    # overall_score_distance = np.mean(score_distance) if len(score_distance) else 0
    # overall_score_angle = np.mean(score_angle) if len(score_angle) else 0
    # final_score = 0.8 * overall_score_distance + 0.2 * overall_score_angle
    #
    # #genarate marks for arms
    # score_distance_arms, score_angle_arms = genarate_marks(distance_array_arms, smooth_angle_arms)
    # overall_score_distance_arms = np.mean(score_distance_arms) if len(score_distance_arms) else 0
    # overall_score_angle_arms = np.mean(score_angle_arms) if len(score_angle_arms) else 0
    # final_score_arms = 0.8 * overall_score_distance_arms + 0.2 * overall_score_angle_arms
    #
    # # genarate marks for legs
    # score_distance_legs, score_angle_legs = genarate_marks(distance_array_legs, smooth_angle_legs)
    # overall_score_distance_legs = np.mean(score_distance_legs) if len(score_distance_legs) else 0
    # overall_score_angle_legs = np.mean(score_angle_legs) if len(score_angle_legs) else 0
    # final_score_legs = 0.8 * overall_score_distance_legs + 0.2 * overall_score_angle_legs
    #
    # # genarate marks for mid_body
    # score_distance_mid, score_angle_mid = genarate_marks(distance_array_mid, smooth_angle_mid)
    # overall_score_distance_mid = np.mean(score_distance_mid) if len(score_distance_mid) else 0
    # overall_score_angle_mid = np.mean(score_angle_mid) if len(score_angle_mid) else 0
    # final_score_mid = 0.8 * overall_score_distance_mid + 0.2 * overall_score_angle_mid
    #
    # print(f"Distance Score = {overall_score_distance:.2f}")
    # print(f"Angle Score    = {overall_score_angle:.2f}")
    # print(f"Final  full body    = {final_score:.2f}")
    # print(f"Final  arms    = {final_score_arms:.2f}")
    # print(f"Final  legs   = {final_score_legs:.2f}")
    # print(f"Final  mid    = {final_score_mid:.2f}")

    # 8) Generate overlay video (sequential)
    generate_overlay_video(
        test_video_path="test_synced.mp4",
        reference_video_path="reference_synced.mp4",
        reference_landmarks_csv="landmarks_with_timestamps_ref.csv",
        test_landmarks_csv="landmarks_with_timestamps_test.csv",
        output_video_path="overlay_video.mp4"
    )

    # generate_glowing_overlay_video_slow(
    #     test_video_path="test_synced.mp4",
    #     reference_video_path="reference_synced.mp4",
    #     reference_landmarks_csv="landmarks_with_timestamps_ref.csv",
    #     test_landmarks_csv="landmarks_with_timestamps_test.csv",
    #     output_video_path="overlay_video.mp4"
    # )

    # Generate 2x2 Matrix Video
    # generate_2x2_matrix_video(
    #     "reference_synced.mp4", "test_synced.mp4",
    #     "landmarks_with_timestamps_ref.csv", "landmarks_with_timestamps_test.csv",
    #     "final_2x2_comparison.mp4",
    #     final_score_arms, final_score_legs, final_score_mid,
    #     max_error_threshold=0.35
    # )
    # scoreScreen("scores.mp4","reference_synced",final_score,final_score_arms,final_score_legs,final_score_mid)
    generate_live_score_video("overlay_video.mp4", "live_score.mp4", "landmarks_with_timestamps_ref.csv",
                              "landmarks_with_timestamps_test.csv", 30)
    generate_side_by_side_heatmap_with_score(
        ref_video="reference_synced.mp4",
        test_video="live_score.mp4",
        ref_csv="landmarks_with_timestamps_ref.csv",
        test_csv="landmarks_with_timestamps_test.csv",
        output_video="side_by_side_video.mp4",
    )
    # plot_distance_vs_frame(distance_array)
