import argparse
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = '/absolute/path/to/face_landmarker.task'

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0, 1, 2, ...)")
    parser.add_argument(
        "--model",
        type=str,
        default="face_landmarker.task",
        help="Path to MediaPipe face_landmarker.task model file",
    )
    args = parser.parse_args()

    print(f"Opening camera index {args.camera} ...")

    # Try AVFoundation backend first on macOS (more reliable)
    try:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    except TypeError:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("Could not open camera. Try a different index, e.g. --camera 1")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # --- MediaPipe Tasks: Face Landmarker setup (VIDEO mode) ---
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    landmarker = FaceLandmarker.create_from_options(options)

    cv2.namedWindow("Eye Tracker", cv2.WINDOW_NORMAL)
    print("Starting face landmark detection... Press 'q' in the video window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)

            # Convert to MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                h, w, _ = frame.shape
                # Use the first detected face
                landmarks = result.face_landmarks[0]

                # Approximate iris indices in Face Landmarker / Face Mesh topology
                # Left eye iris (viewer’s left)
                left_iris_indices = [469, 470, 471, 472]
                # Right eye iris (viewer’s right)
                right_iris_indices = [474, 475, 476, 477]

                # Draw left iris in green
                for idx in left_iris_indices:
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # Draw right iris in cyan
                for idx in right_iris_indices:
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

            cv2.imshow("Eye Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

