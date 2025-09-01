# RealTime Filters System with OpenCV and MediaPipe

This project is a real-time hand gesture-based filter application using OpenCV, MediaPipe, and NumPy. It allows users to select and apply visual effects to a region of the webcam feed by using their hands and finger gestures.

## Features
- **Hand Tracking:** Uses MediaPipe to detect and track up to two hands in real time.
- **Gesture-Based Filter Selection:** Select filters by hovering your index finger over on-screen buttons.
- **Polygonal Region Filtering:** When two hands are detected, a dynamic quadrilateral is formed between thumb and index fingers of both hands. The selected filter is applied only inside this region.
- **Visual Feedback:** Shows filter buttons, tracking visualization, and status text on the video feed.

## Filters
1. **Thermal Filter:** Realistic heat map effect using HOT colormap.
2. **Grayscale Filter:** Grayscale with histogram equalization for better contrast.
3. **High-Contrast Black & White:** CLAHE-based dramatic black and white effect.
4. **Particle Effect:** Random colored particles overlay.
5. **Sepia Filter:** Warm sepia tone using a color transformation kernel.

## How It Works
- **Button Selection:** Hover your index finger over any of the five on-screen buttons to select a filter.
- **Polygonal Filtering:** Show two hands to activate the polygonal region. The filter is applied only inside the quadrilateral formed by thumb and index fingers of both hands.
- **Tracking Visualization:** The bottom half of the window shows green dots and lines for finger tracking and displays the distance between thumb and index fingers.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

- `opencv-python`
- `mediapipe`
- `numpy`

## Usage
Run the script:
```bash
python main.py
```

- Press `ESC` to exit.
- Press `1`-`5` to manually select filters.


## Credits
- Built with [OpenCV](https://opencv.org/), [MediaPipe](https://mediapipe.dev/), and [NumPy](https://numpy.org/)

---
**Author:** Ragul T
