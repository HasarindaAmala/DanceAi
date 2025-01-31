import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm  # <-- for progress bars

# =====================
# Initialize MediaPipe
# =====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose1 = mp_pose.Pose(static_image_mode=False, model_complexity=2,smooth_landmarks=True,min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose2 = mp_pose.Pose(static_image_mode=False,model_complexity=2,smooth_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)


# ============================================================
# 1. Function to smooth (denoise) landmark data
# ============================================================
def smooth_landmarks(landmarks, window_size=5):
    """
    Applies a simple moving-average filter to reduce jitter.
    landmarks is a list of lists: shape (num_frames, 1 + 33*4)
      - First column is timestamp, next 33*4 columns are x,y,z,visibility for each landmark.
    We'll keep timestamps as-is and only smooth the numeric pose data.
    """
    data = np.array(landmarks, dtype=float)  # shape (num_frames, 1 + 33*4)
    smoothed = data.copy()

    # Indices 1..end are the actual pose coords, skip index 0 (timestamp)
    coords_only = data[:, 1:]  # shape (num_frames, 33*4)

    half_w = window_size // 2
    num_frames = len(data)
    for i in range(num_frames):
        start = max(0, i - half_w)
        end = min(num_frames, i + half_w + 1)
        # Average the subset
        smoothed[i, 1:] = np.mean(coords_only[start:end, :], axis=0)

    return smoothed.tolist()


# ========================================================================
# 2. Extract landmarks from a video, save skeleton video & CSV of landmarks
# ========================================================================
def process_video(video_path, pose, output_skeleton_video, output_csv):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_skeleton_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    # 33 Pose Landmarks in MediaPipe
    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye',
        'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky',
        'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip',
        'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
        'right_heel', 'left_foot_index', 'right_foot_index'
    ]

    columns = (
            ['timestamp'] +
            [f'{name}_{coord}' for name in landmark_names for coord in ['x', 'y', 'z', 'visibility']]
    )

    landmarks_data = []
    frame_count = 0

    # Use tqdm to show progress while reading frames
    for _ in tqdm(range(total_frames), desc=f"Processing {video_path}"):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Collect landmark data
            landmarks_frame = []
            for landmark in results.pose_landmarks.landmark:
                landmarks_frame.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            landmarks_data.append([timestamp] + landmarks_frame)

            # Create a black frame and draw skeleton
            blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            mp_drawing.draw_landmarks(blank_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            out.write(blank_frame)

    cap.release()
    out.release()

    df = pd.DataFrame(landmarks_data, columns=columns)
    df.to_csv(output_csv, index=False)

    print(f"Processed video: {video_path}")
    return fps


# ======================================================
# 3. Calculate similarity between two sets of landmarks
# ======================================================
def calculate_similarity(landmarks1, landmarks2):
    # Exclude face landmarks (first 11: nose to mouth_right)
    body_landmarks1 = [frame[11 * 4:] for frame in landmarks1]  # skip first 11 landmarks
    body_landmarks2 = [frame[11 * 4:] for frame in landmarks2]

    similarities = []
    for frame1, frame2 in zip(body_landmarks1, body_landmarks2):
        diff = np.array(frame1) - np.array(frame2)
        frame_distance = np.linalg.norm(diff)
        similarities.append(frame_distance)

    # Example debug: print first 10 similarity values
    print("First 10 similarity values:", similarities[:10])
    print("max similarity values:", max(similarities))
    print("min similarity values:", min(similarities))
    return similarities


# =========================================
# 4. Generate marks (linear distance -> 0-100)
# =========================================
def generate_marks(similarities):
    if not similarities:
        return []
    max_score = max(similarities)
    if max_score == 0:
        max_score = 1
    marks = [100 - (score / max_score * 100) for score in similarities]
    return marks


# ==============================================================================
# 5. Use Local Min/Max for Best/Worst Moments Instead of Simple Sorting
# ==============================================================================
def get_best_and_worst_moments(similarities, top_n, fps, clip_duration):
    """
    1) Compute a sliding window average over clip_duration frames
    2) Find local minima (best) and local maxima (worst)
    3) Pick up to top_n from each category, ensuring no overlap
    """
    clip_frames = int(clip_duration * fps)
    num_frames = len(similarities)

    if num_frames < clip_frames:
        print("Video too short for the given clip_duration.")
        return [], []

    # 1) Compute sliding window averages
    window_averages = []
    for i in range(num_frames - clip_frames + 1):
        window_avg = np.mean(similarities[i: i + clip_frames])
        window_averages.append(window_avg)

    # Helper functions
    def is_local_min(arr, i):
        if i == 0 or i == len(arr) - 1:
            return False
        return (arr[i] < arr[i - 1]) and (arr[i] < arr[i + 1])

    def is_local_max(arr, i):
        if i == 0 or i == len(arr) - 1:
            return False
        return (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1])

    # 2) Find local minima (best) & maxima (worst)
    best_candidates = []
    worst_candidates = []
    for i in range(1, len(window_averages) - 1):
        if is_local_min(window_averages, i):
            best_candidates.append((i, window_averages[i]))
        if is_local_max(window_averages, i):
            worst_candidates.append((i, window_averages[i]))

    # If no local min/max, fallback to absolute min/max
    if not best_candidates:
        min_idx = np.argmin(window_averages)
        best_candidates = [(min_idx, window_averages[min_idx])]
    if not worst_candidates:
        max_idx = np.argmax(window_averages)
        worst_candidates = [(max_idx, window_averages[max_idx])]

    best_candidates.sort(key=lambda x: x[1])  # ascending
    worst_candidates.sort(key=lambda x: x[1], reverse=True)  # descending

    # 3) Pick top_n from each, ensuring no overlap
    def pick_indices(candidates):
        chosen = []
        for idx, _val in candidates:
            if len(chosen) >= top_n:
                break
            if all(abs(idx - c) >= clip_frames for c in chosen):
                chosen.append(idx)
        return chosen

    best_indices = pick_indices(best_candidates)
    worst_indices = pick_indices(worst_candidates)

    print(f"Best Indices: {best_indices}, Worst Indices: {worst_indices}")
    return best_indices, worst_indices


# =========================================================
# 6. Extract Clips with progress bar
# =========================================================
def extract_clips(video_path, indices, output_dir, clip_duration, fps, top_n):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = int(clip_duration * fps)

    extracted_clips = 0
    for i, idx in enumerate(indices):
        if extracted_clips >= top_n:
            break

        start_frame = max(0, idx)
        end_frame = start_frame + clip_frames
        if end_frame > frame_count:
            end_frame = frame_count

        print(f"Processing clip {extracted_clips + 1} at frame {start_frame} -> {end_frame}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = cv2.VideoWriter(
            os.path.join(output_dir, f"clip_{extracted_clips + 1}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), fps,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        # Show progress for frames in this clip
        total_clip_frames = end_frame - start_frame
        for _ in tqdm(range(total_clip_frames), desc=f"Extracting to {output_dir}/clip_{extracted_clips + 1}.mp4"):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        extracted_clips += 1

    cap.release()
    print(f"Extracted {extracted_clips} clips to {output_dir}")


# =========================================================
# 7. Generate Overlay Video with progress bar
# =========================================================
def generate_overlay_video(test_video_path, reference_video_path,
                           reference_landmarks_csv, test_landmarks_csv,
                           output_video_path):
    global scale_factor
    import cv2
    import pandas as pd
    import mediapipe as mp

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

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    # Estimate an initial scale factor from up to 50 frames
    scale_factor = 1.0
    sample_frames = min(50, len(landmarks_df), len(landmarks_test))
    for i in range(sample_frames):
        ref_row = landmarks_df.iloc[i]
        test_row = landmarks_test.iloc[i]
        width_test_temp = abs(test_row['right_shoulder_x'] - test_row['left_shoulder_x'])
        width_ref_temp = abs(ref_row['right_shoulder_x'] - ref_row['left_shoulder_x'])
        if width_ref_temp < 1e-6:
            width_ref_temp = 1e-6
        scale_factor = width_test_temp / width_ref_temp

    scale_factor = max(0.8, min(scale_factor, 1.2))
    mp_pose_module = mp.solutions.pose
    pose_connections = mp_pose_module.POSE_CONNECTIONS

    frame_count = 0

    # Use tqdm to visualize progress while building overlay
    for _ in tqdm(range(total_frames), desc=f"Generating overlay -> {output_video_path}"):
        ret, frame = cap.read()
        if not ret or frame_count >= len(landmarks_df):
            break

        ref_row = landmarks_df.iloc[frame_count]
        test_row = landmarks_test.iloc[frame_count]

        # Calculate upper body centers dynamically
        ref_upper_body_x = (
                                   ref_row['right_shoulder_x'] +
                                   ref_row['left_shoulder_x'] +
                                   ref_row['right_hip_x'] +
                                   ref_row['left_hip_x']
                           ) / 4
        ref_upper_body_y = (
                                   ref_row['right_shoulder_y'] +
                                   ref_row['left_shoulder_y'] +
                                   ref_row['right_hip_y'] +
                                   ref_row['left_hip_y']
                           ) / 4

        test_upper_body_x = (
                                    test_row['right_shoulder_x'] +
                                    test_row['left_shoulder_x'] +
                                    test_row['right_hip_x'] +
                                    test_row['left_hip_x']
                            ) / 4
        test_upper_body_y = (
                                    test_row['right_shoulder_y'] +
                                    test_row['left_shoulder_y'] +
                                    test_row['right_hip_y'] +
                                    test_row['left_hip_y']
                            ) / 4

        # Calculate offsets
        x_offset = (test_upper_body_x * frame_width) - (ref_upper_body_x * frame_width_ref * scale_factor)
        y_offset = (test_upper_body_y * frame_height) - (ref_upper_body_y * frame_height_ref * scale_factor)

        # Reshape reference landmarks
        ref_landmarks = ref_row[1:].values.reshape(-1, 4)  # x, y, z, visibility

        # Draw reference skeleton
        for connection in pose_connections:
            start_idx, end_idx = connection
            if ref_landmarks[start_idx][3] > 0.5 and ref_landmarks[end_idx][3] > 0.5:
                x1 = int((ref_landmarks[start_idx][0] * frame_width_ref) * scale_factor + x_offset)
                y1 = int((ref_landmarks[start_idx][1] * frame_height_ref) * scale_factor + y_offset)
                x2 = int((ref_landmarks[end_idx][0] * frame_width_ref) * scale_factor + x_offset)
                y2 = int((ref_landmarks[end_idx][1] * frame_height_ref) * scale_factor + y_offset)
                # Slightly thicker line behind for better visibility

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw test upper body center (blue circle)
        test_center_x = int(test_upper_body_x * frame_width)
        test_center_y = int(test_upper_body_y * frame_height)
        cv2.circle(frame, (test_center_x, test_center_y), 10, (255, 0, 0), -1)

        # Draw aligned reference upper body center (red circle)
        ref_center_x = int(ref_upper_body_x * frame_width_ref * scale_factor + x_offset)
        ref_center_y = int(ref_upper_body_y * frame_height_ref * scale_factor + y_offset)
        cv2.circle(frame, (ref_center_x, ref_center_y), 10, (0, 0, 255), -1)

        out.write(frame)
        frame_count += 1

    cap.release()
    cap_ref.release()
    out.release()
    print(f"Overlay video saved to {output_video_path}")


# =========================================================
# 8. Main Execution
# =========================================================
if __name__ == "__main__":
    # Process reference and test videos -> CSV (with progress bars)
    fps_ref = process_video('reference.mp4', pose1, 'skeleton_video1.mp4', 'landmarks_with_timestamps_1.csv')
    fps_test = process_video('test.mp4', pose2, 'skeleton_video2.mp4', 'landmarks_with_timestamps_2.csv')

    # Load landmarks data as lists
    raw_landmarks1 = pd.read_csv('landmarks_with_timestamps_1.csv').values.tolist()
    raw_landmarks2 = pd.read_csv('landmarks_with_timestamps_2.csv').values.tolist()

    # 1) Smooth the landmark data to reduce jitter
    landmarks1 = smooth_landmarks(raw_landmarks1, window_size=5)
    landmarks2 = smooth_landmarks(raw_landmarks2, window_size=5)

    # 2) Calculate similarities
    similarities = calculate_similarity(landmarks1, landmarks2)

    # 3) Generate marks (0..100) and overall score
    marks = generate_marks(similarities)
    overall_score = np.mean(marks) if marks else 0
    print(f"Overall Score: {overall_score:.2f}")

    # 4) Overlay video (progress bar included)
    generate_overlay_video('test.mp4', 'reference.mp4',
                           'landmarks_with_timestamps_1.csv',
                           'landmarks_with_timestamps_2.csv',
                           'overlay_video.mp4')

    # 5) Identify best and worst moments
    top_n = 5
    clip_duration = 2  # seconds
    best_indices, worst_indices = get_best_and_worst_moments(similarities, top_n, fps_ref, clip_duration)

    # 6) Extract best and worst clips (progress bars included)
    extract_clips('overlay_video.mp4', best_indices, 'best_clips', clip_duration, fps_ref, top_n)
    extract_clips('overlay_video.mp4', worst_indices, 'worst_clips', clip_duration, fps_ref, top_n)

    print("Best and worst moment clips saved successfully.")