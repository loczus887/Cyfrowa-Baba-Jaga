import cv2
import numpy as np
import time
import pygame

class Timeit:
    """
    Dekorator do mierzenia czasu wykonania funkcji.
    """
    @staticmethod
    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Czas wykonania funkcji {func.__name__}: {end - start} sekund.")
            return result
        return wrapper

class MotionDetector:
    """
    Klasa do detekcji ruchu.
    """
    def __init__(self):
        self.prev_gray = None

    @Timeit.timeit
    def detect_motion(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            return False, []
        
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, frame_diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        frame_diff_thresh = cv2.morphologyEx(frame_diff_thresh, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        contours, _ = cv2.findContours(frame_diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                motion_detected = True
                break
        
        self.prev_gray = gray.copy()
        return motion_detected, contours

    def bounding_box(self, contours):
        motion_box = None
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                motion_box = (x, y, w, h)
                break
        return motion_box

class YOLODetector:
    """
    Klasa do detekcji ludzi za pomocą YOLO.
    """
    def __init__(self, yolo_weights, yolo_cfg, coco_names):
        self.net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        with open(coco_names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    @Timeit.timeit
    def detect_people(self, frame, motion_box):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        people_detected = False
        
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                motion_center_x = motion_box[0] + motion_box[2] // 2
                motion_center_y = motion_box[1] + motion_box[3] // 2
                if x <= motion_center_x <= x + w and y <= motion_center_y <= y + h:
                    people_detected = True
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), cv2.FILLED)
        
        return people_detected, frame

class SoundPlayer:
    """
    Klasa do odtwarzania dźwięku.
    """
    @staticmethod
    def play_sound(file_path):
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

class DigitalBabaYaga:
    """
    Główna klasa do obsługi aplikacji Cyfrowa Baba Jaga.
    """
    def __init__(self):
        self.warp_points = []
        self.motion_detector = MotionDetector()
        self.yolo_detector = YOLODetector('yolov3.weights', 'yolov3.cfg', 'coco.names')
        self.cap = cv2.VideoCapture(0)
        self.last_time_spoken = time.time()
        self.current_time_spoken = time.time()
        
        cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Motion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('Motion Detection', self.on_mouse_click)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.warp_points) < 4:
                self.warp_points.append((x, y))
                self.update_frame()
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            for point in self.warp_points:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)
            cv2.imshow('Motion Detection', frame)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break

            if len(self.warp_points) < 4:
                cv2.imshow('Motion Detection', frame)
                cv2.waitKey(1)
                continue

            h, w = frame.shape[:2]
            pts1 = np.float32(self.warp_points)
            pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(frame, matrix, (w, h))

            self.current_time_spoken = time.time()

            if self.current_time_spoken - self.last_time_spoken <= 5:
                SoundPlayer.play_sound("cyfrowa_bj.mp3")
                time.sleep(5)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            elif self.current_time_spoken - self.last_time_spoken > 6 and self.current_time_spoken - self.last_time_spoken <= 7:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion_detected, contours = self.motion_detector.detect_motion(gray)
                result = cv2.warpPerspective(frame, matrix, (w, h))
                cv2.imshow('Motion Detection', result)
                
            elif self.current_time_spoken - self.last_time_spoken > 7 and self.current_time_spoken - self.last_time_spoken <= 13:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion_detected, contours = self.motion_detector.detect_motion(gray)

                if motion_detected:
                    motion_box = self.motion_detector.bounding_box(contours)
                    people_detected, frame_with_box = self.yolo_detector.detect_people(frame, motion_box)

                    if people_detected:
                        result = cv2.warpPerspective(frame_with_box, matrix, (w, h))
                        cv2.imshow('Motion Detection', result)
            
            elif self.current_time_spoken - self.last_time_spoken > 13 and self.current_time_spoken - self.last_time_spoken < 14:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion_detected, contours = self.motion_detector.detect_motion(gray)
                result = cv2.warpPerspective(frame, matrix, (w, h))
                cv2.imshow('Motion Detection', result)

            elif self.current_time_spoken - self.last_time_spoken > 14:
                self.current_time_spoken = time.time()
                self.last_time_spoken = self.current_time_spoken
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DigitalBabaYaga()
    app.run()
