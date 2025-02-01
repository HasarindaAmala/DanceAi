import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm  # <-- for progress bars

from vocal import VocalSync
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose1 = mp_pose.Pose(static_image_mode=False, model_complexity=2,smooth_landmarks=True,min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose2 = mp_pose.Pose(static_image_mode=False,model_complexity=2,smooth_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)



def preProcess(ref_path,Test_path):

    #get both fps and make it equal
    #sync the sound and place test video on the ref video
    obj_vocal = VocalSync(ref_path,Test_path)
    ref_sync,test_sync = obj_vocal.start_vocalSync()
    #trim and make both have same duration

    #return two videos(ref_sync,test_sync)
    return ref_sync,test_sync

def generate_csv(
    video_path,
    pose,
    output_skeleton_video,
    output_abs_csv,
    output_diff_csv
):
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

    # Columns for the absolute CSV
    abs_columns = (
        ['timestamp'] +
        [f'{name}_{coord}' for name in landmark_names for coord in ['x', 'y', 'z', 'visibility']]
    )

    # Columns for the diffs CSV
    # We'll have [frame_no] plus 3 columns per landmark: diff_x, diff_y, slope
    diff_columns = (
        ['frame_no'] +
        [f"{name}_diff_x" for name in landmark_names] +
        [f"{name}_diff_y" for name in landmark_names] +
        [f"{name}_slope" for name in landmark_names]
    )

    # We'll store absolute landmark data here
    landmarks_data = []
    # We'll store per-frame diffs for all landmarks here
    diffs_data = []

    # We'll keep track of the previous frame's (x, y) for each landmark in a list or array
    prev_xy = None  # shape: (33, 2) once assigned

    frame_count = 0

    for _ in tqdm(range(total_frames), desc=f"Processing {video_path}"):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Collect *absolute* landmark data
            frame_landmarks = []
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            # Save to absolute CSV buffer
            landmarks_data.append([timestamp] + frame_landmarks)

            # ====== Compute diffs for all landmarks ======
            # We'll build a row: [frame_no] + [ all diff_x ] + [ all diff_y ] + [ all slope ]
            diff_row = [frame_count - 1]  # store frame_no first

            # We'll create arrays to temporarily hold each landmark's diff_x, diff_y, slope
            all_diff_x = []
            all_diff_y = []
            all_slope  = []

            # Reshape current (x,y,z,vis) for convenience
            current_xy = []
            for i in range(33):
                x_val = frame_landmarks[i*4 + 0]
                y_val = frame_landmarks[i*4 + 1]
                current_xy.append((x_val, y_val))
            current_xy = np.array(current_xy)  # shape = (33,2)

            if prev_xy is None:
                # If there's no previous frame, we treat diffs as 0
                # or you could skip storing for this first frame
                diff_x_array = np.zeros(33, dtype=float)
                diff_y_array = np.zeros(33, dtype=float)
                slope_array = np.zeros(33, dtype=float)
            else:
                diff_x_array = current_xy[:,0] - prev_xy[:,0]
                diff_y_array = current_xy[:,1] - prev_xy[:,1]
                slope_array = np.zeros(33, dtype=float)
                for i in range(33):
                    dx = diff_x_array[i]
                    dy = diff_y_array[i]
                    if abs(dx) < 1e-10:
                        slope_array[i] = 0
                    else:
                        slope_array[i] = dy / dx

            # Now append them in the correct order:
            # first all diff_x, then all diff_y, then all slope
            diff_row.extend(diff_x_array.tolist())
            diff_row.extend(diff_y_array.tolist())
            diff_row.extend(slope_array.tolist())

            # store this row in diffs_data
            diffs_data.append(diff_row)
            # update prev_xy
            prev_xy = current_xy

            # Create a black frame and draw skeleton
            blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            mp_drawing.draw_landmarks(blank_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            out.write(blank_frame)

    cap.release()
    out.release()

    # Create the absolute CSV
    df_abs = pd.DataFrame(landmarks_data, columns=abs_columns)
    df_abs.to_csv(output_abs_csv, index=False)

    # Create the diffs CSV
    df_diff = pd.DataFrame(diffs_data, columns=diff_columns)
    df_diff.to_csv(output_diff_csv, index=False)

    print(f"Processed video: {video_path}")
    print(f"Absolute CSV saved to: {output_abs_csv}")
    print(f"Diff CSV saved to: {output_diff_csv}")

    return fps



def calculate_similarity_diff_csv(ref_csv, test_csv):
    """
    Compare two 'diff CSV' files from 33 landmarks.
    Each file has columns:
      frame_no,
      [33 columns of _diff_x],
      [33 columns of _diff_y],
      [33 columns of _slope].

    We'll produce TWO metrics:
     1) distance-based difference: L2 norm of (dx_ref - dx_test, dy_ref - dy_test) across all 33 landmarks
     2) angle-based difference: L2 norm of (arctan(slope_ref) - arctan(slope_test)) across 33 landmarks

    Returns:
     dist_diff_array: shape (n_frames,), distance-based difference per frame
     angle_diff_array: shape (n_frames,), angle-based difference per frame
     dist_score: single float, e.g. average of dist_diff_array
     angle_score: single float, e.g. average of angle_diff_array
    """

    df_ref = pd.read_csv(ref_csv)
    df_test = pd.read_csv(test_csv)

    # Ensure both have the same number of rows
    min_len = min(len(df_ref), len(df_test))
    df_ref = df_ref.iloc[:min_len]
    df_test = df_test.iloc[:min_len]

    # Layout of columns:
    #   0: frame_no
    #   1..33:  _diff_x  for 33 landmarks
    #   34..66: _diff_y  for 33 landmarks
    #   67..99: _slope   for 33 landmarks
    dx_start = 1
    dx_end = dx_start + 33  # 1..33
    dy_start = dx_end
    dy_end = dy_start + 33  # 34..66
    slope_start = dy_end  # 67
    slope_end = slope_start + 33  # 100

    n_frames = min_len
    dist_diff_array = np.zeros(n_frames, dtype=float)
    angle_diff_array = np.zeros(n_frames, dtype=float)

    for i in range(n_frames):
        row_ref = df_ref.iloc[i].values
        row_test = df_test.iloc[i].values

        # Extract arrays for the 33 landmarks
        ref_dx = row_ref[dx_start: dx_end]  # shape (33,)
        ref_dy = row_ref[dy_start: dy_end]  # shape (33,)
        ref_slope = row_ref[slope_start: slope_end]  # shape (33,)

        test_dx = row_test[dx_start: dx_end]
        test_dy = row_test[dy_start: dy_end]
        test_slope = row_test[slope_start: slope_end]

        # 1) Distance-based difference
        # For each landmark j: we get diff_dx_j = ref_dx[j]-test_dx[j], same for diff_dy_j
        diff_dx = ref_dx - test_dx
        diff_dy = ref_dy - test_dy
        # We can either sum across j in some manner. For a single measure:
        # Flatten them together: [diff_dx(0..32), diff_dy(0..32)] => shape (66,)
        diff_dist = np.concatenate([diff_dx, diff_dy])  # shape (66,)
        # Then do an L2 norm
        dist_diff = np.linalg.norm(diff_dist)

        dist_diff_array[i] = dist_diff

        # 2) Angle-based difference
        angle_ref = np.arctan(ref_slope)
        angle_test = np.arctan(test_slope)
        diff_angle = angle_ref - angle_test  # shape (33,)

        # L2 norm across all 33 landmarks
        angle_diff = np.linalg.norm(diff_angle)

        angle_diff_array[i] = angle_diff


    return dist_diff_array, angle_diff_array


def draw_graph(diff_csv):
    print()
    #draw graph (y = test angle,ref angle,diff | x = frame_no)

    #calculate starting time of dance (filter out diff)

    #return starting_time, end_time

def genarate_marks(similarities,angle):

    max_score_dis = max(similarities)
    max_score_angle = max(angle)
    if max_score_dis == 0:
        max_score_dis = 1
    if max_score_angle == 0:
        max_score_angle = 1
    marks_dis = [100 - (score / max_score_dis * 100) for score in similarities]
    marks_angle = [100 - (score / max_score_angle * 100) for score in angle]

    return marks_dis,marks_angle


# =========================================================
# 8. Main Execution
# =========================================================
if __name__ == "__main__":

    ref_sync,test_sync = preProcess("reference.mp4","test.mp4",)
    ref_csv = generate_csv("reference_synced.mp4", pose1, 'skeleton_video_ref.mp4', 'landmarks_with_timestamps_ref.csv','diff_landmarks_ref.csv')
    test_csv = generate_csv("test_synced.mp4", pose2, 'skeleton_video_test.mp4', 'landmarks_with_timestamps_test.csv','diff_landmarks_test.csv')

    distance,angle = calculate_similarity_diff_csv("diff_landmarks_ref.csv", 'diff_landmarks_test.csv')
    #starting_time,end_time = draw_graph(similarities_csv)
    score_distance,score_angle = genarate_marks(distance,angle)
    overall_score_distance = np.mean(score_distance) if score_distance else 0
    overall_score_angle = np.mean(score_angle) if score_angle else 0
    print(overall_score_distance)
    print(overall_score_angle)
