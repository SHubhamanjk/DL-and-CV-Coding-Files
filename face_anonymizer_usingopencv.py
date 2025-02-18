import os
import cv2
import mediapipe as mp

def preprocess_input(img=None, video=None, webcam=False):
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

        if img is not None:
            image = cv2.imread(img)
            if image is None:
                print("Error: Unable to read the image file.")
                return
            processed_img = do_blur(image, face_detection)
            cv2.imshow('Image', processed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif video is not None:
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                print("Error: Unable to open the video file.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error reading frame.")
                    break

                frame = do_blur(frame, face_detection)
                if frame is not None:
                    cv2.imshow('Video', frame)
                else:
                    print("Error: Frame not processed properly.")
                    break

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        elif webcam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Unable to access the webcam.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from webcam.")
                    break

                frame = do_blur(frame, face_detection)
                if frame is not None:
                    cv2.imshow('Webcam', frame)
                else:
                    print("Error: Frame not processed properly.")
                    break

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

def do_blur(img, face_detection):
    if img is None:
        print("Error: Received an empty image.")
        return img

    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = max(0, int(x1 * W))
            y1 = max(0, int(y1 * H))
            w = int(w * W)
            h = int(h * H)

            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            if x1 < x2 and y1 < y2:
                blur_size = (min(w, 100), min(h, 100))
                if blur_size[0] > 0 and blur_size[1] > 0:
                    img[y1:y2, x1:x2] = cv2.blur(img[y1:y2, x1:x2], blur_size)

    return img


def face_anonymizer(img=None, video=None, webcam=False):
    if img is not None:
        preprocess_input(img)
    elif video is not None:
        preprocess_input(video=video)
    else:
        preprocess_input(webcam=True)

face_anonymizer(webcam=True)
