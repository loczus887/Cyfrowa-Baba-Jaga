# project name: "Digital Baba Yaga"
# authors: Alicja Kowalska, Klaudia Karczmarczyk
# nazwa projektu: "Cyfrowa Baba Jaga"
# autorzy: Alicja Kowalska, Klaudia Karczmarczyk

import cv2
import numpy as np
from collections import deque
import time
import pygame
import time

warp_points = []

def timeit(func):
    """
    Dekorator do mierzenia czasu wykonania funkcji.

    Argumenty:
    func -- funkcja, której czas wykonania jest mierzony

    Zwraca:
    wrapper -- funkcja opakowująca, która mierzy czas wykonania
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Czas wykonania funkcji {func.__name__}: {end - start} sekund.")
        return result
    return wrapper


def on_mouse_click(event, x, y, flags, param):
    """
    Obsługa zdarzeń kliknięcia myszy. Dodaje punkty kliknięcia do listy warp_points.

    Argumenty:
    event -- typ zdarzenia myszy
    x -- współrzędna x kliknięcia
    y -- współrzędna y kliknięcia
    flags -- dodatkowe flagi związane ze zdarzeniem
    param -- dodatkowe parametry przekazywane do funkcji (nieużywane)
    """
    global warp_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(warp_points) < 4:
            warp_points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  
            cv2.imshow('Motion Detection', frame)  
        if len(warp_points) == 4:
            pass

# Wczytaj pre-trenowany model YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#Wczytaj etykiety klas
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

@timeit
def detect_motion(gray, prev_gray):
    """
    Detekcja ruchu między dwoma klatkami.

    Argumenty:
    gray -- aktualna klatka w odcieniach szarości
    prev_gray -- poprzednia klatka w odcieniach szarości

    Zwraca:
    motion_detected -- flaga wskazująca czy wykryto ruch
    contours -- lista konturów wykrytych w klatce
    """
    # Obliczanie różnicy między dwoma klatkami
    frame_diff = cv2.absdiff(prev_gray, gray)
    # Progowanie różnicy
    _, frame_diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    # Usuwanie szumów
    frame_diff_thresh = cv2.morphologyEx(frame_diff_thresh, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    # Znajdowanie konturów
    contours, _ = cv2.findContours(frame_diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sprawdzanie czy kontury spełniają warunki ruchu
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            motion_detected = True
            break
    return motion_detected, contours

#@timeit
def bounding_box(contours, result):
    """
    Wyznacza prostokąt ograniczający dla wykrytego ruchu.

    Argumenty:
    contours -- lista konturów
    result -- klatka obrazu

    Zwraca:
    motion_box -- współrzędne prostokąta ograniczającego dla wykrytego ruchu
    """
    motion_box = None
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            motion_box = (x, y, w, h)
            break  # Zatrzymujemy się po wykryciu pierwszego obszaru z ruchem
    return motion_box


def point_inside_rect(x, y, rect):
    """
    Sprawdza, czy punkt (x, y) znajduje się wewnątrz prostokąta o danych współrzędnych i wymiarach.

    Argumenty:
    x -- współrzędna x punktu
    y -- współrzędna y punktu
    rect -- współrzędne prostokąta (x, y, szerokość, wysokość)

    Zwraca:
    True jeśli punkt znajduje się wewnątrz prostokąta, w przeciwnym razie False
    """
    rx, ry, rw, rh = rect
    if x >= rx and x <= rx + rw and y >= ry and y <= ry + rh:
        return True
    else:
        return False

@timeit
def detect_people_yolo(result):
    """
    Detekcja ludzi za pomocą YOLO w danym obszarze.

    Argumenty:
    result -- obraz wynikowy po przekształceniu perspektywy

    Zwraca:
    people_detected -- flaga wskazująca, czy wykryto ludzi
    frame -- klatka obrazu z zaznaczonymi prostokątami wykrytych ludzi
    """
    blob = cv2.dnn.blobFromImage(result, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    boxes = []
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID for 'person' is 0
                # Współrzędne prostokąta
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
             # Obliczenie środka obszaru ruchu
            motion_center_x = motion_box[0] + motion_box[2] // 2
            motion_center_y = motion_box[1] + motion_box[3] // 2
            # Sprawdzenie czy środek obszaru ruchu znajduje się wewnątrz obszaru detekcji ludzi
            if x <= motion_center_x <= x + w and y <= motion_center_y <= y + h:
                people_detected = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), cv2.FILLED)  # Rysuj czerwony prostokąt               
    return people_detected, frame 

def play_sound(file_path):
    """
    Odtwarza dźwięk z pliku.

    Argumenty:
    file_path -- ścieżka do pliku dźwiękowego
    """
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

# Initializacja kamery
cap = cv2.VideoCapture(0)
cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Motion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('Motion Detection', on_mouse_click)

# Zmienna do śledzenia czasu odtwarzania dźwięku
last_time_spoken = time.time()   
current_time_spoken = time.time()

# Główna pętla programu
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    # Przechwytuj kliknięcia myszą, aby uzyskać punkty do przekształcenia
    if len(warp_points) < 4:
        cv2.imshow('Motion Detection', frame)
        cv2.waitKey(1)
        continue

    # Wykonaj przekształcenie perspektywistyczne
    h, w = frame.shape[:2]
    pts1 = np.float32(warp_points)
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (w, h))

    current_time_spoken = time.time()

    #Sprawdzenie, czy minęło 5 sekund od ostatniego odtworzenia
    if current_time_spoken - last_time_spoken <= 5:
       
        mp3_file = "cyfrowa_bj.mp3"  # Zmień nazwę pliku MP3
        play_sound(mp3_file)
        time.sleep(5)  # Odtwarzaj dźwięk przez 5 sekund
        # Konwersja klatki do odcieni szarości
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
    elif current_time_spoken - last_time_spoken > 6 and current_time_spoken - last_time_spoken <= 7:
        # Konwersja klatki do odcieni szarości
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detekcja ruchu
        if 'prev_gray' not in locals():
            prev_gray = gray.copy()
        motion_detected, contours = detect_motion(gray, prev_gray)
        prev_gray = gray.copy()
        result = cv2.warpPerspective(frame, matrix, (w, h))
        cv2.imshow('Motion Detection', result)
    elif current_time_spoken - last_time_spoken > 7 and current_time_spoken - last_time_spoken <= 13:
        # Konwersja klatki do odcieni szarości
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detekcja ruchu
        if 'prev_gray' not in locals():
            prev_gray = gray.copy()
        motion_detected, contours = detect_motion(gray, prev_gray)
        prev_gray = gray.copy()

        if motion_detected:
            motion_box = bounding_box(contours, frame)
            people_detected, frame_with_box = detect_people_yolo(frame)

            if people_detected:
                # Wyświetlenie klatki z rysunkami
                result = cv2.warpPerspective(frame_with_box, matrix, (w, h))
                cv2.imshow('Motion Detection', result)

    elif current_time_spoken - last_time_spoken > 13 and current_time_spoken - last_time_spoken <14:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detekcja ruchu
        if 'prev_gray' not in locals():
            prev_gray = gray.copy()
        motion_detected, contours = detect_motion(gray, prev_gray)
        prev_gray = gray.copy()
        result = cv2.warpPerspective(frame, matrix, (w, h))
        cv2.imshow('Motion Detection', result)

    elif current_time_spoken - last_time_spoken > 14:
        current_time_spoken = time.time()
        last_time_spoken = current_time_spoken
        
    # Sprawdzenie czy użytkownik nacisnął klawisz 'q', jeśli tak - przerwanie pętli
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów kamery i zamknięcie okien
cap.release()
cv2.destroyAllWindows()