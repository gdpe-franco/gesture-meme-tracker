"""
Gesture Meme Tracker - Real-time Hand Gesture Detection with Meme Display
Uses MediaPipe Hands to detect gestures and displays corresponding meme images
"""

import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands and Face
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Dictionary mapping gestures to cat-meme image/video filenames
GESTURE_MEMES = {
    "chill": "cat-reactions/chill.png",
    "confident": "cat-reactions/confident.jpg",
    "desperate": "cat-reactions/desperate.jpeg",
    "disgusted": "cat-reactions/disgusted.jpeg",
    "happy": "cat-reactions/happy.gif",
    "heart": "cat-reactions/heart.jpg",
    "mad": "cat-reactions/mad.png",
    "pointing_front": "cat-reactions/pointing-front.jpeg",
    "pointing_up": "cat-reactions/pointing-up.jpg",
    "punch": "cat-reactions/punch.png",
    "sad": "cat-reactions/sad.jpeg",
    "scared": "cat-reactions/scared.jpg",
    "scuba": "cat-reactions/scuba.gif",
    "thumbs_up": "cat-reactions/thumbs-up.jpg",
    "tongue_out": "cat-reactions/tongue-out.png",
    "none": "cat-reactions/confident.jpg",
}


def detect_gesture_legacy(hand_landmarks, all_hands=None, face_landmarks=None):
    """
    Detects hand gesture based on landmark positions.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object containing 21 points
        all_hands: List of all detected hands (for two-hand gestures)
        
    Returns:
        String representing the detected gesture name
        
    Landmark indices (key points):
    - 0: Wrist
    - 4: Thumb tip
    - 8: Index finger tip
    - 12: Middle finger tip
    - 16: Ring finger tip
    - 20: Pinky tip
    - 3, 7, 11, 15, 19: Finger DIPs (second from tip)
    - 2, 6, 10, 14, 18: Finger PIPs (middle joints)
    """
    
    # Extract landmark coordinates
    landmarks = hand_landmarks.landmark
    
    # Helper function to check if finger is extended
    def is_finger_extended(tip_idx, pip_idx, mcp_idx):
        """Check if a finger is extended by comparing y-coordinates"""
        # In image coordinates, y increases downward, so tip.y < pip.y means extended
        return landmarks[tip_idx].y < landmarks[pip_idx].y and landmarks[pip_idx].y < landmarks[mcp_idx].y
    
    # Count extended fingers (excluding thumb)
    extended_fingers = []
    
    # Index finger (8: tip, 6: PIP, 5: MCP)
    if is_finger_extended(8, 6, 5):
        extended_fingers.append("index")
    
    # Middle finger (12: tip, 10: PIP, 9: MCP)
    if is_finger_extended(12, 10, 9):
        extended_fingers.append("middle")
    
    # Ring finger (16: tip, 14: PIP, 13: MCP)
    if is_finger_extended(16, 14, 13):
        extended_fingers.append("ring")
    
    # Pinky finger (20: tip, 18: PIP, 17: MCP)
    if is_finger_extended(20, 18, 17):
        extended_fingers.append("pinky")
    
    num_extended = len(extended_fingers)
    
    # Get key landmarks
    wrist = landmarks[0]
    index_tip = landmarks[8]
    
    # CUSTOM GESTURES
    
    # JIJIJA - Just laughing (mouth open detection only)
    if face_landmarks:
        # Key face landmarks for mouth detection
        # 13: upper lip, 14: lower lip, 61: mouth corner, 84: mouth corner
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        left_corner = face_landmarks.landmark[61]
        right_corner = face_landmarks.landmark[84]
        
        # Calculate mouth opening
        mouth_height = abs(upper_lip.y - lower_lip.y)
        mouth_width = abs(right_corner.x - left_corner.x)
        
        # Check if mouth is open (laughing) - very sensitive thresholds
        if mouth_height > 0.01 and mouth_width > 0.005:  # Very sensitive thresholds for open mouth
            return "jijija"
    
    # MIMIMI - Both hands closed (fists) anywhere
    if all_hands and len(all_hands) == 2:
        # Check if both hands have no fingers extended (closed fists)
        hand1_extended = 0
        hand2_extended = 0
        
        # Count extended fingers for hand 1
        for tip_idx, pip_idx, mcp_idx in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]:
            if all_hands[0].landmark[tip_idx].y < all_hands[0].landmark[pip_idx].y:
                hand1_extended += 1
        
        # Count extended fingers for hand 2
        for tip_idx, pip_idx, mcp_idx in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]:
            if all_hands[1].landmark[tip_idx].y < all_hands[1].landmark[pip_idx].y:
                hand2_extended += 1
        
        # Both hands closed (no fingers extended)
        if hand1_extended == 0 and hand2_extended == 0:
            return "mimimi"
    
    # CERRAO - One hand closed but one finger (index) extended
    if num_extended == 1 and "index" in extended_fingers:
        return "cerrao"
    
    # SIXSEVEN - Hands extended like doing a balance
    # Both hands with open palms, extended outward
    if all_hands and len(all_hands) == 2:
        # Check if both hands have extended fingers
        hand1_extended = sum([
            all_hands[0].landmark[8].y < all_hands[0].landmark[6].y,
            all_hands[0].landmark[12].y < all_hands[0].landmark[10].y,
            all_hands[0].landmark[16].y < all_hands[0].landmark[14].y,
        ])
        hand2_extended = sum([
            all_hands[1].landmark[8].y < all_hands[1].landmark[6].y,
            all_hands[1].landmark[12].y < all_hands[1].landmark[10].y,
            all_hands[1].landmark[16].y < all_hands[1].landmark[14].y,
        ])
        
        if hand1_extended >= 2 and hand2_extended >= 2:
            # Check if hands are spread apart (like balancing)
            hand1_wrist = all_hands[0].landmark[0]
            hand2_wrist = all_hands[1].landmark[0]
            x_distance = abs(hand1_wrist.x - hand2_wrist.x)
            if x_distance > 0.3:  # Hands far apart
                return "sixseven"
    
    # Default - no gesture detected
    return "none"


def detect_gesture(hand_landmarks, all_hands=None, face_landmarks=None):
    """Match visible hand poses to the supplied cat reactions."""
    landmarks = hand_landmarks.landmark

    def extended(tip, pip, mcp):
        return landmarks[tip].y < landmarks[pip].y < landmarks[mcp].y

    fingers = {
        "index": extended(8, 6, 5),
        "middle": extended(12, 10, 9),
        "ring": extended(16, 14, 13),
        "pinky": extended(20, 18, 17),
    }
    open_fingers = sum(fingers.values())
    thumb_extended = abs(landmarks[4].x - landmarks[5].x) > 0.08

    if all_hands and len(all_hands) == 2:
        open_counts = [sum(hand.landmark[tip].y < hand.landmark[pip].y
                           for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)])
                       for hand in all_hands]
        wrists = [hand.landmark[0] for hand in all_hands]
        if face_landmarks:
            mouth = face_landmarks.landmark[13]
            covered = [abs(hand.landmark[9].x - mouth.x) < 0.14 and
                       abs(hand.landmark[9].y - mouth.y) < 0.14
                       for hand in all_hands]
            raised = [wrist.y < 0.45 for wrist in wrists]
            raised_y = next((wrists[index].y for index in range(2)
                             if not covered[index] and raised[index]), None)
            previous_y = getattr(detect_gesture, "scuba_hand_y", raised_y)
            detect_gesture.scuba_hand_y = raised_y
            if any(covered) and raised_y is not None and abs(raised_y - previous_y) > 0.012:
                return "scuba"
        index_tips_close = abs(all_hands[0].landmark[8].x - all_hands[1].landmark[8].x) < 0.08
        thumb_tips_close = abs(all_hands[0].landmark[4].x - all_hands[1].landmark[4].x) < 0.10
        if index_tips_close and thumb_tips_close:
            return "heart"
        if all(count >= 3 for count in open_counts):
            if all(wrist.y < 0.45 for wrist in wrists):
                return "desperate"
            return "scared"
        if all(count == 0 for count in open_counts):
            return "mad"

    if open_fingers == 0:
        return "punch"
    if thumb_extended and fingers["pinky"] and open_fingers == 1:
        return "chill"
    if thumb_extended and open_fingers == 0:
        return "thumbs_up"
    if fingers["index"] and open_fingers == 1:
        if getattr(landmarks[8], "z", 0) < getattr(landmarks[6], "z", 0) - 0.08:
            return "pointing_front"
        return "pointing_up"
    if fingers["index"] and fingers["middle"] and open_fingers == 2:
        return "disgusted"
    if face_landmarks:
        face = face_landmarks.landmark
        mouth_height = abs(face[13].y - face[14].y)
        mouth_width = abs(face[61].x - face[291].x)
        mouth_center = (face[13].y + face[14].y) / 2
        if abs(face[61].y - face[291].y) > 0.012:
            return "confident"
        if face[61].y > mouth_center + 0.012 and face[291].y > mouth_center + 0.012:
            return "sad"
        if mouth_height > 0.025:
            return "tongue_out"
        if mouth_width > 0.12:
            return "happy"
    return "none"


def load_meme_media(images_folder):
    """
    Load all meme images and videos from the specified folder.
    
    Args:
        images_folder: Path to folder containing meme images/videos
        
    Returns:
        Tuple of (media_dict, video_caps_dict, is_video_dict):
        - media_dict: Images or first frame of videos
        - video_caps_dict: VideoCapture objects for videos
        - is_video_dict: Boolean flags indicating if media is video
    """
    meme_images = {}
    video_caps = {}
    is_video = {}
    
    for gesture, filename in GESTURE_MEMES.items():
        media_path = os.path.join(images_folder, filename)
        
        # Check if it's a video file
        if filename.endswith(('.mp4', '.avi', '.mov', '.webm', '.gif')):
            is_video[gesture] = True
            if os.path.exists(media_path):
                cap = cv2.VideoCapture(media_path)
                if cap.isOpened():
                    video_caps[gesture] = cap
                    # Read first frame for display
                    ret, frame = cap.read()
                    if ret:
                        meme_images[gesture] = frame
                    else:
                        meme_images[gesture] = create_placeholder_image(gesture)
                else:
                    meme_images[gesture] = create_placeholder_image(gesture)
            else:
                meme_images[gesture] = create_placeholder_image(gesture)
        else:
            # It's an image file
            is_video[gesture] = False
            if os.path.exists(media_path):
                img = cv2.imread(media_path)
                if img is not None:
                    meme_images[gesture] = img
                else:
                    meme_images[gesture] = create_placeholder_image(gesture)
            else:
                meme_images[gesture] = create_placeholder_image(gesture)
    
    return meme_images, video_caps, is_video


def create_placeholder_image(gesture_name):
    """
    Create a placeholder image with text when meme image is not found.
    
    Args:
        gesture_name: Name of the gesture
        
    Returns:
        Numpy array representing the placeholder image
    """
    # Create a colored background
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Different colors for different gestures
    colors = {
        "jijija": (0, 165, 255),        # Orange
        "mimimi": (203, 192, 255),      # Pink
        "sixseven": (255, 144, 30),     # Blue
        "cerrao": (147, 20, 255),       # Purple
        "none": (128, 128, 128)         # Gray
    }
    
    color = colors.get(gesture_name, (255, 255, 255))
    img[:] = color
    
    # Add text
    text = gesture_name.upper().replace("_", " ")
    cv2.putText(img, text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 0), 3, cv2.LINE_AA)
    
    return img


def resize_meme(meme_image, target_height):
    """
    Resize meme image to match the target height while maintaining aspect ratio.
    
    Args:
        meme_image: Original meme image
        target_height: Desired height in pixels
        
    Returns:
        Resized image
    """
    height, width = meme_image.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    
    # Resize image
    resized = cv2.resize(meme_image, (new_width, target_height), 
                        interpolation=cv2.INTER_AREA)
    
    return resized


def main():
    """
    Main function to run the Gesture Meme Tracker application.
    """
    print("=" * 60)
    print("Cat Gesture Meme Tracker")
    print("Press 'q' to quit")
    print("=" * 60)
    print("\nGestures: shaka, thumbs up, point up/front, fist, heart, and two-hand poses")
    print("Face reactions: happy, tongue out, and scuba/surprised")
    print("\nMake sure your hand(s) are visible to the camera!")
    print("=" * 60)
    
    # Create images folder if it doesn't exist
    images_folder = os.path.join(os.path.dirname(__file__), "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        print(f"\nCreated images folder at: {images_folder}")
        print("Note: Add your meme images to this folder!")
        print("Expected filenames:", list(GESTURE_MEMES.values()))
        print()
    
    # Load meme images and videos
    meme_images, video_caps, is_video = load_meme_media(images_folder)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    # Set camera resolution (optional, adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize MediaPipe Hands and Face
    with mp_hands.Hands(
        static_image_mode=False,          # False for video stream
        max_num_hands=2,                   # Detect up to two hands
        min_detection_confidence=0.7,      # Confidence threshold for detection
        min_tracking_confidence=0.5        # Confidence threshold for tracking
    ) as hands, mp_face_mesh.FaceMesh(
        static_image_mode=False,          # False for video stream
        max_num_faces=1,                  # Detect one face
        refine_landmarks=True,            # Refine landmarks for better accuracy
        min_detection_confidence=0.5,     # Confidence threshold for detection
        min_tracking_confidence=0.5       # Confidence threshold for tracking
    ) as face_mesh:
        
        current_gesture = "none"
        
        while True:
            # Read frame from webcam
            success, frame = cap.read()
            
            if not success:
                print("Failed to grab frame from webcam!")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Get frame dimensions
            frame_height, frame_width, _ = frame.shape
            
            # Convert BGR to RGB (MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands and Face
            hand_results = hands.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)
            
            # Check if face detected and draw landmarks
            face_landmarks = None
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                # Draw face landmarks (optional, can comment out for cleaner view)
                mp_drawing.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                )
            
            # Check if hand(s) detected
            if hand_results.multi_hand_landmarks:
                # Draw all hand landmarks
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                
                # Detect gesture (pass all hands and face for multi-hand gestures)
                current_gesture = detect_gesture(
                    hand_results.multi_hand_landmarks[0],
                    all_hands=hand_results.multi_hand_landmarks,
                    face_landmarks=face_landmarks
                )
            elif face_landmarks:
                # No hands detected but face is visible - check for face-only gestures like JIJIJA
                # Create a dummy hand landmark for the function call
                dummy_landmarks = type('obj', (object,), {'landmark': [type('obj', (object,), {'x': 0, 'y': 0})] * 21})()
                current_gesture = detect_gesture(
                    dummy_landmarks,
                    all_hands=None,
                    face_landmarks=face_landmarks
                )
            else:
                # No hands or face detected, reset to neutral
                current_gesture = "none"
            
            # Get appropriate meme for current gesture
            # If it's a video, read the next frame
            if is_video.get(current_gesture, False) and current_gesture in video_caps:
                ret, video_frame = video_caps[current_gesture].read()
                if ret:
                    meme_images[current_gesture] = video_frame
                else:
                    # Loop video from beginning
                    video_caps[current_gesture].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, video_frame = video_caps[current_gesture].read()
                    if ret:
                        meme_images[current_gesture] = video_frame
            
            meme = meme_images.get(current_gesture, meme_images["none"])
            
            # Resize meme to match frame height
            meme_resized = resize_meme(meme, frame_height)
            
            # Ensure meme width doesn't exceed reasonable size
            meme_height, meme_width = meme_resized.shape[:2]
            if meme_width > frame_width:
                meme_resized = cv2.resize(meme_resized, (frame_width, frame_height))
                meme_width = frame_width
            
            # Create combined display (webcam + meme side by side)
            combined_width = frame_width + meme_width
            combined_frame = np.zeros((frame_height, combined_width, 3), dtype=np.uint8)
            
            # Place webcam frame on the left
            combined_frame[:, :frame_width] = frame
            
            # Place meme on the right
            combined_frame[:, frame_width:frame_width + meme_width] = meme_resized
            
            # Add text overlay showing current gesture
            gesture_text = current_gesture.replace("_", " ").title()
            cv2.putText(combined_frame, f"Gesture: {gesture_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Debug: Show mouth values if face is detected
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                upper_lip = face_landmarks.landmark[13]
                lower_lip = face_landmarks.landmark[14]
                left_corner = face_landmarks.landmark[61]
                right_corner = face_landmarks.landmark[84]
                
                mouth_height = abs(upper_lip.y - lower_lip.y)
                mouth_width = abs(right_corner.x - left_corner.x)
                
                cv2.putText(combined_frame, f"Mouth H: {mouth_height:.3f} W: {mouth_width:.3f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add instructions
            cv2.putText(combined_frame, "Press 'q' to quit", 
                       (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Display the combined frame
            cv2.imshow('Cat Gesture Meme Tracker', combined_frame)
            
            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting Gesture Meme Tracker...")
                break
    
    # Release resources
    cap.release()
    
    # Release all video captures
    for video_cap in video_caps.values():
        if video_cap.isOpened():
            video_cap.release()
    
    cv2.destroyAllWindows()
    print("Application closed successfully!")


if __name__ == "__main__":
    main()
