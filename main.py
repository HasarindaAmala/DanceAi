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
    if max_score == 0:
        max_score = 1
    marks = [100 - (score / max_score * 100) for score in similarities]
    return marks


def get_best_and_worst_moments(similarities, top_n, fps, clip_duration):
    clip_frames = clip_duration * fps  # Total number of frames in a clip
    num_frames = len(similarities)

    # Compute sliding window averages for similarities
    window_averages = []
    for i in range(num_frames - clip_frames + 1):
        window_avg = np.mean(similarities[i:i + clip_frames])
        window_averages.append(window_avg)

    # Sort indices based on the window averages
    sorted_indices = np.argsort(window_averages)

    # Select the best indices (smallest averages) and worst indices (largest averages)
    best_indices = []
    worst_indices = []

    best_indices.append(sorted_indices[0])  # Add the smallest similarity index
    worst_indices.append(sorted_indices[-1])  # Add the largest similarity index

    for n in range(1, top_n):
        # Find the next valid best index
        for i in sorted_indices:
            if all(abs(i - prev_idx) >= clip_frames for prev_idx in best_indices):  # Check non-overlap
                best_indices.append(i)
                break

        # Find the next valid worst index
        for j in sorted_indices[::-1]:
            if all(abs(j - prev_idx) >= clip_frames for prev_idx in worst_indices):  # Check non-overlap
                worst_indices.append(j)
                break

    print(f"Best Indices: {best_indices}, Worst Indices: {worst_indices}")
    return best_indices, worst_indices


def extract_clips(video_path, indices, output_dir, clip_duration, fps):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = clip_duration * fps  # Total number of frames in a clip

    extracted_clips = 0
    for i, idx in enumerate(indices):
        print(f"Processing clip {i + 1} at frame {idx}")
        if extracted_clips >= 3:  # Limit to 3 clips
            break

        start_frame = max(0, idx)
        end_frame = start_frame + clip_frames

        # Ensure the clip duration fits within the video length
        if end_frame > frame_count:
            end_frame = frame_count

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = cv2.VideoWriter(
            f'{output_dir}/clip_{extracted_clips + 1}.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'), fps,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        extracted_clips += 1

    cap.release()
    print(f"Extracted {extracted_clips} clips to {output_dir}")
def generate_overlay_video(test_video_path, reference_video_path, reference_landmarks_csv, test_landmarks_csv, output_video_path):
    global scale_factor
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

    for i in range(50):
        if i >= len(landmarks_df) or i >= len(landmarks_test):
            break

        time_row = landmarks_df.iloc[i]
        time_row_test = landmarks_test.iloc[i]

        # Get width values for test and reference
        width_test_temp = abs(time_row_test['right_shoulder_x'] - time_row_test['left_shoulder_x'])
        width_ref_temp = abs(time_row['right_shoulder_x'] - time_row['left_shoulder_x'])
        scale_factor = width_test_temp/width_ref_temp
    # Constrain the scaling factor dynamically
    scale_factor = max(0.8, min(scale_factor, 1.2))
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



        # Calculate offsets to align dynamically
        x_offset = (test_upper_body_x * frame_width) - (ref_upper_body_x * frame_width_ref * scale_factor)
        y_offset = (test_upper_body_y * frame_height) - (ref_upper_body_y * frame_height_ref * scale_factor)

        #print(f"Frame {frame_count}: Scale Factor = {scale_factor}, X Offset = {x_offset}, Y Offset = {y_offset}")

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

    # Generate overlay video
    generate_overlay_video('test.mp4', 'reference.mp4', 'landmarks_with_timestamps_1.csv',
                           'landmarks_with_timestamps_2.csv', 'overlay_video.mp4')
    # Get best and worst moments
    best_indices, worst_indices = get_best_and_worst_moments(similarities,3,fps1,3)

    # Extract clips
    extract_clips('overlay_video.mp4', best_indices, 'best_clips', 3, fps2)
    extract_clips('overlay_video.mp4', worst_indices, 'worst_clips',3, fps2)



    print("Best and worst moment clips saved successfully.")
