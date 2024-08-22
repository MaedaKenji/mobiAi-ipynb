import cv2
import mediapipe as mp
import os
import numpy as np
import time
from datetime import datetime
from PIL import Image

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5)

# Create a directory for saving images if it doesn't exist
save_directory = "C:\\TA\\trainCNN\\30cm"
sKelas = "Kanan"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

cap = cv2.VideoCapture(0)

def get_combined_bounding_box(landmarks, img_width, img_height):
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    x_min, x_max = min(x_coords) * img_width, max(x_coords) * img_width
    y_min, y_max = min(y_coords) * img_height, max(y_coords) * img_height
    return int(x_min), int(y_min), int(x_max), int(y_max)

prev_frame_time = 0  # Initialize the previous frame time variable

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # continue # Original code
        exit()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time
    fps_text = f'FPS: {fps:.2f}'
            
    # Process the frame with MediaPipe FaceMesh
    image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    
    # Prepare a black background
    black_image = np.zeros(image.shape, dtype=np.uint8)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Eyes Landmarks
            LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
            RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

            # Iris Landmarks
            LEFT_IRIS = [474, 475, 476, 477]
            RIGHT_IRIS = [469, 470, 471, 472]
            MID_LEFT_IRIS = [473]
            MID_RIGHT_IRIS = [468]

            # Combine all into 2 separate array
            all_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE + RIGHT_EYE + LEFT_IRIS + RIGHT_IRIS]
            
            # Normalize Landmark Value
            mesh_points = np.array(
                [
                        np.multiply([i.x, i.y], [w, h]).astype(int)
                        for i in results.multi_face_landmarks[0].landmark
                ]
            )

            # Draw Eyelid Landmarks on Mask
            cv2.polylines(black_image, [mesh_points[LEFT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
            cv2.polylines(black_image, [mesh_points[RIGHT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
            
            # Draw Iris Landmarks on Mask
            cv2.polylines(black_image, [mesh_points[LEFT_IRIS]], True, (255,255,255), 1, cv2.LINE_AA)
            cv2.polylines(black_image, [mesh_points[RIGHT_IRIS]], True, (255,255,255), 1, cv2.LINE_AA)

            # Calculate the combined bounding box for both eyes
            combined_x_min, combined_y_min, combined_x_max, combined_y_max = get_combined_bounding_box(all_eye_landmarks, w, h)
            
            # Crop the combined eye region from the black background image
            combined_eye_image = black_image[combined_y_min:combined_y_max, combined_x_min:combined_x_max]
            
            # Save the image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{timestamp}.jpg"
            combined_eye_path = os.path.join(save_directory, sKelas, filename)
            # cv2.imwrite(combined_eye_path, combined_eye_image) # Original Code
            
            
            print(f"Cropped image saved at: {combined_eye_path}")
            
            # Optionally display the cropped combined eye image (for testing)
            try:
                cv2.imshow('Combined Eyes', combined_eye_image)

            except cv2.error as e:
                print("Eyes not found")

        # Put FPS text on the frame
        cv2.putText(image, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Break the loop on 'q' key press
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()