# README

## Project Name: "Digital Baba Yaga"
## Authors: Alicja Kowalska, Klaudia Karczmarczyk

---

## Project Overview

"Digital Baba Yaga" is a Python-based application that utilizes computer vision and deep learning techniques to detect motion and identify people using a webcam. The project leverages the YOLO (You Only Look Once) object detection model and OpenCV for real-time image processing and detection.

## Installation and Setup

### Requirements
- Python 3.x
- OpenCV
- NumPy
- Pygame
- YOLO model files (`yolov3.weights`, `yolov3.cfg`, and `coco.names`)

### Installation
1. Clone the repository or download the project files.
2. Ensure you have Python 3.x installed on your system.
3. Install the required Python libraries:
   ```bash
   pip install opencv-python numpy pygame
   ```
4. Download the YOLO model files (`yolov3.weights`, `yolov3.cfg`) and the class labels (`coco.names`). Place these files in the project directory.

## Usage

1. Run the script:
   ```bash
   python script_name.py
   ```
2. A window named "Motion Detection" will open, displaying the webcam feed.
3. Click four points on the window to define the region for perspective transformation.
4. The system will start detecting motion and people within the defined region.

### Key Functionality
- **Mouse Click Handling:** The user defines the region of interest by clicking four points on the initial frame.
- **Motion Detection:** Detects motion between consecutive frames.
- **YOLO People Detection:** Uses the YOLO model to detect people within the motion-detected region.
- **Sound Playback:** Plays a sound file if motion is detected within the specified intervals.

## Code Explanation

### Main Components

1. **timeit Decorator:**
   Measures and prints the execution time of the decorated function.

2. **on_mouse_click Function:**
   Handles mouse click events to capture four points for perspective transformation.

3. **detect_motion Function:**
   Detects motion between two grayscale frames using frame differencing, thresholding, and contour detection.

4. **bounding_box Function:**
   Determines the bounding box for the detected motion.

5. **point_inside_rect Function:**
   Checks if a point is inside a given rectangle.

6. **detect_people_yolo Function:**
   Detects people using the YOLO model within the transformed perspective.

7. **play_sound Function:**
   Plays a sound file using the Pygame library.

### Main Loop
- Captures frames from the webcam.
- Performs perspective transformation using user-defined points.
- Detects motion and people based on specific time intervals.
- Plays a sound file when certain conditions are met.
- Displays the processed frames in real-time.

### Notes
- The sound file (`cyfrowa_bj.mp3`) should be placed in the project directory.
- The script operates in fullscreen mode and can be exited by pressing the 'q' key.

