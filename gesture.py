import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.9,
                       min_tracking_confidence=0.9,
                       max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def detect_gesture(hand_landmarks):
    """
    Returns a string representing the gesture based on finger states.
    Finger states are determined as follows:
    - For fingers (index, middle, ring, pinky): if the tip landmark's y-coordinate 
      is less than that of the PIP joint (tip-2), the finger is considered "up".
    - For the thumb, a simple check using the x-coordinates (suitable for mirror view) is used.
    """
    # Create a list for finger states: True means finger is raised.
    # Order: [thumb, index, middle, ring, pinky]
    fingers = []
    
    # Thumb: For mirror view, we check if thumb tip is to the right of thumb IP joint.
    thumb_up = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
    fingers.append(thumb_up)
    
    # For index, middle, ring, and pinky fingers:
    for tip in [8, 12, 16, 20]:
        # Finger is up if tip is higher (i.e., smaller y) than PIP joint (tip-2)
        finger_up = hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y
        fingers.append(finger_up)
    
    # Calculate Euclidean distance between thumb tip and index finger tip for OK gesture
    thumb_index_dist = np.sqrt((hand_landmarks.landmark[4].x - hand_landmarks.landmark[8].x) ** 2 +
                               (hand_landmarks.landmark[4].y - hand_landmarks.landmark[8].y) ** 2)
    
    # Define gestures based on finger states and distance:
    # Victory Gesture: Index and middle fingers up, ring and pinky down (thumb doesn't matter)
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "✌ Victory"
    
    # OK Gesture: Thumb and index touching (distance below threshold) 
    # (other fingers can be ignored for this gesture)
    if thumb_index_dist < 0.05:
        return "👌 OK"
    
    # Fist: All fingers down
    if not any(fingers):
        return "✊ Fist"
    
    # Open Palm: All fingers up
    if all(fingers):
        return "✋ Open Palm"
    
    # Thumbs Up: Only thumb up
    if fingers[0] and not any(fingers[1:]):
        return "👍 Thumbs Up"
    
    return "❓ Unknown"

def perform_action(gesture):
    """
    Define actions based on the recognized gesture.
    You can modify these actions as needed.
    """
    if gesture == "👍 Thumbs Up":
        pyautogui.press("volumeup")  # Increase volume
    elif gesture == "✊ Fist":
        pyautogui.press("volumemute")  # Mute/Unmute volume
    elif gesture == "✋ Open Palm":
        pyautogui.rightClick()         # Right-click
    elif gesture == "✌ Victory":
        pyautogui.click()              # Left-click
    elif gesture == "👌 OK":
        pyautogui.press("enter")       # Press Enter

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mirror and resize the frame for faster processing
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    
    # Convert BGR frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    gesture = "No Hand Detected"
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            perform_action(gesture)
            cv2.putText(frame, gesture, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, gesture, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()