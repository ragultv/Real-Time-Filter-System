import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Hands Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Filters
def apply_filter(frame, filter_type):
    if filter_type == 1:  # Improved thermal filter (using HOT colormap for more realistic heat map)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)  # HOT for better thermal effect
        return thermal
    elif filter_type == 2:  # Grayscale (unchanged, but added equalization for better contrast)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    elif filter_type == 3:  # High-contrast Black and White like the given image (using CLAHE for enhanced contrast)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))  # Higher clipLimit for more dramatic contrast
        high_contrast = clahe.apply(gray)
        return cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)
    elif filter_type == 4:  # Particle effect filter (random colored particles on the original frame)
        particle_frame = frame.copy()
        h, w, _ = particle_frame.shape
        num_particles = 100
        particle_positions = np.random.randint(0, [w, h], size=(num_particles, 2))
        for pos in particle_positions:
            color = tuple(np.random.randint(0, 256, 3).tolist())  # Random color for particles
            cv2.circle(particle_frame, tuple(pos), 2, color, -1)
        return particle_frame
    elif filter_type == 5:  # Improved Sepia (with clipping to prevent overflow and added warmth adjustment)
        kernel = np.array([[0.272, 0.543, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return sepia
    return frame

def apply_filter_to_polygon(frame, filter_type, polygon_points):
    """Apply filter only to the specified polygon region"""
    if polygon_points is None or len(polygon_points) < 3:
        return frame
    
    # Create a mask for the polygon
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    polygon_points = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon_points], 255)
    
    # Apply filter to the entire frame
    filtered_frame = apply_filter(frame, filter_type)
    
    # Create result by combining original and filtered using mask
    result = frame.copy()
    result[mask == 255] = filtered_frame[mask == 255]
    
    return result

def get_thumb_index_points(hand_landmarks, w, h):
    """Get only thumb tip and index finger tip points"""
    # Thumb tip (landmark 4)
    thumb_tip = (int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h))
    
    # Index finger tip (landmark 8)
    index_tip = (int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h))
    
    return thumb_tip, index_tip

def create_dynamic_quadrilateral(hand1_points, hand2_points):
    """Create a dynamic quadrilateral from two hands' thumb and index finger points"""
    if hand1_points is None or hand2_points is None:
        return None
    
    hand1_thumb, hand1_index = hand1_points
    hand2_thumb, hand2_index = hand2_points
    
    # Create quadrilateral using the four points
    # Order points to create a proper quadrilateral
    points = [hand1_thumb, hand1_index, hand2_index, hand2_thumb]
    
    return points

def create_single_hand_polygon(hand_points, expansion_factor=50):
    """Create a polygon around single hand's thumb and index finger"""
    if hand_points is None:
        return None
    
    thumb_tip, index_tip = hand_points
    
    # Calculate distance between thumb and index
    distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
    
    # Create a dynamic polygon based on finger distance
    # When fingers are close, polygon is smaller; when far, polygon is larger
    expansion = max(20, min(expansion_factor, distance * 0.5))
    
    # Calculate center point
    center_x = (thumb_tip[0] + index_tip[0]) // 2
    center_y = (thumb_tip[1] + index_tip[1]) // 2
    
    # Create diamond/rhombus shape around the center
    polygon_points = [
        (center_x, center_y - int(expansion)),  # Top
        (center_x + int(expansion * 0.8), center_y),  # Right
        (center_x, center_y + int(expansion)),  # Bottom
        (center_x - int(expansion * 0.8), center_y)   # Left
    ]
    
    return polygon_points

def check_button_hover(finger_pos, button_positions):
    """Check if finger is hovering over any button"""
    for i, (bx, by) in enumerate(button_positions):
        if bx-30 < finger_pos[0] < bx+30 and by-30 < finger_pos[1] < by+30:
            return i + 1  # Return filter number (1-5)
    return None

# Button positions
filters = ["1", "2", "3", "4", "5"]
button_positions = [(80 + i*130, 440) for i in range(5)]

selected_filter = 1  # Start with filter 1
apply_mode = False
polygon_points = None
hand1_points = None
hand2_points = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Create a copy for drawing
    display_frame = frame.copy()
    
    # Create tracking visualization area (bottom half)
    tracking_area = np.zeros((h//2, w, 3), dtype=np.uint8)
    
    # Detect hands
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_positions = []
    index_finger_positions = []
    detected_hands = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get only thumb and index finger points
            thumb_pos, index_pos = get_thumb_index_points(hand_landmarks, w, h)
            detected_hands.append((thumb_pos, index_pos))
            
            # Store index finger position for button selection
            index_finger_positions.append(index_pos)
            
            # Store hand position for tracking visualization
            hand_positions.append((thumb_pos, index_pos))

    # Update hand points for filtering - ONLY when TWO hands are detected
    if len(detected_hands) >= 2:
        hand1_points = detected_hands[0]
        hand2_points = detected_hands[1]
        polygon_points = create_dynamic_quadrilateral(hand1_points, hand2_points)
        apply_mode = True
    else:
        # No filter when less than 2 hands
        apply_mode = False
        polygon_points = None
        hand1_points = None
        hand2_points = None

    # Check for button hover and selection
    for finger_pos in index_finger_positions:
        hovered_filter = check_button_hover(finger_pos, button_positions)
        if hovered_filter:
            selected_filter = hovered_filter
            break

    # Draw filter buttons
    for i, (bx, by) in enumerate(button_positions):
        color = (200, 200, 200)
        if i + 1 == selected_filter:
            color = (0, 255, 0)  # Green for selected
        cv2.circle(display_frame, (bx, by), 30, color, -1)
        cv2.putText(display_frame, filters[i], (bx-10, by+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Apply filter to polygon if active (no visual annotations on main frame)
    if apply_mode and polygon_points:
        display_frame = apply_filter_to_polygon(display_frame, selected_filter, polygon_points)

    # Draw tracking visualization (bottom half)
    # Add green dots and lines for tracking visualization
    if hand_positions:
        for hand_pos in hand_positions:
            if hand_pos[0] is not None and hand_pos[1] is not None:
                thumb_pos, index_pos = hand_pos
                # Scale positions for tracking area
                track_x1 = int(thumb_pos[0] * (w / w))
                track_y1 = int((thumb_pos[1] - h//2) * ((h//2) / (h//2))) if thumb_pos[1] > h//2 else 10
                track_x2 = int(index_pos[0] * (w / w))
                track_y2 = int((index_pos[1] - h//2) * ((h//2) / (h//2))) if index_pos[1] > h//2 else 10
                
                # Ensure points are within tracking area bounds
                track_y1 = max(0, min(h//2 - 1, track_y1))
                track_y2 = max(0, min(h//2 - 1, track_y2))
                
                # Draw green dots and lines in tracking area
                cv2.circle(tracking_area, (track_x1, track_y1), 8, (0, 255, 0), -1)
                cv2.circle(tracking_area, (track_x2, track_y2), 8, (0, 255, 0), -1)
                cv2.line(tracking_area, (track_x1, track_y1), (track_x2, track_y2), (0, 255, 0), 3)
                
                # Calculate and display distance
                distance = np.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
                cv2.putText(tracking_area, f"Distance: {int(distance)}px", 
                           (track_x1 + 20, track_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Combine main frame with tracking area
    combined_frame = np.vstack([display_frame, tracking_area])
    
    # Add status text
    hands_count = len(detected_hands) if detected_hands else 0
    status_text = f"Filter {selected_filter} - {'Active' if apply_mode else 'Inactive'} - Hands: {hands_count}"
    cv2.putText(combined_frame, status_text, (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if apply_mode and polygon_points:
        if len(detected_hands) == 2:
            cv2.putText(combined_frame, "Two-hand filter active", (10, h + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Gesture Controlled Filters", combined_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('1'):
        selected_filter = 1
    elif key == ord('2'):
        selected_filter = 2
    elif key == ord('3'):
        selected_filter = 3
    elif key == ord('4'):
        selected_filter = 4
    elif key == ord('5'):
        selected_filter = 5

cap.release()
cv2.destroyAllWindows()