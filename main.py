import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose1 = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose2 = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract landmarks and save skeleton videos
def process_video(video_path, pose, output_skeleton_video, output_csv):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_skeleton_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye',
        'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky',
        'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip',
        'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
        'right_heel', 'left_foot_index', 'right_foot_index'
    ]

    columns = ['timestamp'] + [f'{name}_{coord}' for name in landmark_names for coord in ['x', 'y', 'z', 'visibility']]

    landmarks_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks_frame = []
            for landmark in results.pose_landmarks.landmark:
                landmarks_frame.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            landmarks_data.append([timestamp] + landmarks_frame)

            blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            mp_drawing.draw_landmarks(blank_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            out.write(blank_frame)

    cap.release()
    out.release()

    df = pd.DataFrame(landmarks_data, columns=columns)
    df.to_csv(output_csv, index=False)

    print(f"Processed video: {video_path}")
    return fps

# Calculate similarity between two sets of landmarks
def calculate_similarity(landmarks1, landmarks2):
    # Exclude face landmarks (first 11 landmarks: nose to mouth_right)
    body_landmarks1 = [frame[11 * 4:] for frame in landmarks1]  # Skip first 11 landmarks
    body_landmarks2 = [frame[11 * 4:] for frame in landmarks2]

    similarities = []
    for frame1, frame2 in zip(body_landmarks1, body_landmarks2):
        diff = np.array(frame1) - np.array(frame2)
        frame_distance = np.linalg.norm(diff)
        similarities.append(frame_distance)
    return similarities

# Generate marks based on similarity
def generate_marks(similarities):
    max_score = max(similarities)
    marks = [100 - (score / max_score * 100) for score in similarities]
    return marks

# Get best and worst moments
def get_best_and_worst_moments(similarities, top_n=3):
    sorted_indices = np.argsort(similarities)
    best_indices = sorted_indices[:top_n]
    worst_indices = sorted_indices[-top_n:]
    return best_indices, worst_indices

# Extract 3-second clips for best and worst moments
def extract_clips(video_path, indices, output_dir, clip_duration=3, fps=30):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    for i, frame_idx in enumerate(indices):
        start_frame = max(0, frame_idx - (clip_duration // 2) * fps)
        end_frame = start_frame + clip_duration * fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = cv2.VideoWriter(
            f'{output_dir}/clip_{i+1}.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'), fps,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()

    cap.release()

def generate_overlay_video(test_video_path, reference_video_path, reference_landmarks_csv, test_landmarks_csv, output_video_path):
    import cv2
    import pandas as pd
    import mediapipe as mp

    # Load videos and landmarks
    cap = cv2.VideoCapture(test_video_path)
    cap_ref = cv2.VideoCapture(reference_video_path)
    landmarks_df = pd.read_csv(reference_landmarks_csv)
    landmarks_test = pd.read_csv(test_landmarks_csv)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    mp_pose = mp.solutions.pose
    pose_connections = mp_pose.POSE_CONNECTIONS

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= len(landmarks_df):
            break

        # Get landmarks for the current frame
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

        # Calculate shoulder widths dynamically
        ref_shoulder_width = abs(ref_row['right_shoulder_x'] - ref_row['left_shoulder_x'])
        test_shoulder_width = abs(test_row['right_shoulder_x'] - test_row['left_shoulder_x'])

        # Calculate scaling factor for this frame
        scale_factor = test_shoulder_width / (ref_shoulder_width + 1e-6)

        # Constrain the scaling factor dynamically
        scale_factor = max(0.8, min(scale_factor, 1.2))

        # Calculate offsets to align dynamically
        x_offset = (test_upper_body_x * frame_width) - (ref_upper_body_x * frame_width_ref * scale_factor)
        y_offset = (test_upper_body_y * frame_height) - (ref_upper_body_y * frame_height_ref * scale_factor)

        print(f"Frame {frame_count}: Scale Factor = {scale_factor}, X Offset = {x_offset}, Y Offset = {y_offset}")

        # Reshape reference landmarks
        ref_landmarks = ref_row[1:].values.reshape(-1, 4)  # x, y, z, visibility

        # Draw reference skeleton dynamically aligned and scaled to the test skeleton
        for connection in pose_connections:
            start_idx, end_idx = connection
            if (
                ref_landmarks[start_idx][3] > 0.5 and  # Visibility check
                ref_landmarks[end_idx][3] > 0.5
            ):
                # Scale and align reference landmarks dynamically
                x1 = int((ref_landmarks[start_idx][0] * frame_width_ref) * scale_factor + x_offset)
                y1 = int((ref_landmarks[start_idx][1] * frame_height_ref) * scale_factor + y_offset)
                x2 = int((ref_landmarks[end_idx][0] * frame_width_ref) * scale_factor + x_offset)
                y2 = int((ref_landmarks[end_idx][1] * frame_height_ref) * scale_factor + y_offset)

                # Draw skeleton connection
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw test upper body center (blue circle)
        test_center_x = int(test_upper_body_x * frame_width)
        test_center_y = int(test_upper_body_y * frame_height)
        cv2.circle(frame, (test_center_x, test_center_y), 10, (255, 0, 0), -1)  # Blue circle

        # Draw aligned reference upper body center (red circle)
        ref_center_x = int(ref_upper_body_x * frame_width_ref * scale_factor + x_offset)
        ref_center_y = int(ref_upper_body_y * frame_height_ref * scale_factor + y_offset)
        cv2.circle(frame, (ref_center_x, ref_center_y), 10, (0, 0, 255), -1)  # Red circle

        # Write the frame
        out.write(frame)
        frame_count += 1

    cap.release()
    cap_ref.release()
    out.release()
    print(f"Overlay video saved to {output_video_path}")
# Main process
if __name__ == "__main__":
    # Process reference and test videos
    fps1 = process_video('reference.mp4', pose1, 'skeleton_video1.mp4', 'landmarks_with_timestamps_1.csv')
    fps2 = process_video('test.mp4', pose2, 'skeleton_video2.mp4', 'landmarks_with_timestamps_2.csv')

    # Load landmarks data
    landmarks1 = pd.read_csv('landmarks_with_timestamps_1.csv').values.tolist()
    landmarks2 = pd.read_csv('landmarks_with_timestamps_2.csv').values.tolist()

    # Calculate similarities
    similarities = calculate_similarity(landmarks1, landmarks2)

    # Generate marks
    marks = generate_marks(similarities)
    print(f"Overall Score: {np.mean(marks):.2f}")

    # Get best and worst moments
    best_indices, worst_indices = get_best_and_worst_moments(similarities)

    # Extract clips
    extract_clips('test.mp4', best_indices, 'best_clips', clip_duration=3, fps=fps2)
    extract_clips('test.mp4', worst_indices, 'worst_clips', clip_duration=3, fps=fps2)

    # Generate overlay video
    generate_overlay_video('test.mp4', 'reference.mp4','landmarks_with_timestamps_1.csv','landmarks_with_timestamps_2.csv', 'overlay_video.mp4')

    print("Best and worst moment clips saved successfully.")
