import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import geocoder  # To get the GPS coordinates (install with 'pip install geocoder')

class SafetyAlertSystem:
    def __init__(self):
        # Load pre-trained model and configuration for gender detection
        gender_model_path = "/Users/tharun/PycharmProjects/women_safety/gender/gender_net.caffemodel"
        gender_proto_path = "/Users/tharun/PycharmProjects/women_safety/gender/gender_deploy.prototxt"
        self.gender_net = cv2.dnn.readNet(gender_model_path, gender_proto_path)

        # List of gender classes
        self.gender_list = ['Male', 'Female']

        # Initialize Mediapipe Hands and Drawing modules
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2)  # Detect up to two hands
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize face detection using OpenCV's pre-trained model (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Open the default webcam
        self.cap = cv2.VideoCapture(0)

        # Folder to save captured images
        self.save_directory = "captured_incidents"
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def get_current_location(self):
        """Get the current GPS location."""
        g = geocoder.ip('me')
        if g.ok:
            return g.latlng  # Returns latitude and longitude as [lat, lng]
        else:
            return None

    def detect_gestures(self, frame, results, genders_detected):
        """Detect hand gestures using Mediapipe and trigger an alert if a woman raises her hand, considering context awareness."""

        # Check if a woman has been detected in the frame
        if 'Female' not in genders_detected:
            return  # No woman detected, no need to process gestures

        # Count the number of men and women
        men_count = genders_detected.count('Male')
        women_count = genders_detected.count('Female')

        # Analyze context: Count number of people in the scene
        num_people = len(genders_detected)

        # Check if the number of people exceeds a certain threshold (e.g., 2)
        if num_people < 2:
            context = "Low Context Awareness"  # Not enough people to assess context
        else:
            context = "High Context Awareness"  # Sufficient number of people for context analysis

        # Process hand gestures if a woman is detected
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Get wrist and index finger coordinates
            wrist = hand_landmarks.landmark[0]
            index_tip = hand_landmarks.landmark[8]

            # Convert the normalized coordinates to pixel values
            height, width, _ = frame.shape
            wrist_x, wrist_y = int(wrist.x * width), int(wrist.y * height)
            index_tip_x, index_tip_y = int(index_tip.x * width), int(index_tip.y * height)

            # Check if the hand is raised
            if wrist.y > index_tip.y:
                gesture = "Raised Hand"

                # Trigger alert and capture image with context awareness, including men and women count
                self.trigger_alert_and_save_image(frame, "Hand Raised Alert by Woman", context, men_count, women_count)
            else:
                gesture = "Hand Down"

            # Display the identified gesture on the screen
            cv2.putText(frame, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    def detect_faces_and_gender(self, frame):
        """Detect faces in the frame and classify gender using the gender model."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        genders_detected = []
        face_locations = []

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face_locations.append((x, y, w, h))

            # Preprocess the face for gender classification
            blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227),
                                         mean=(78.4263377603, 87.7689143744, 114.895847746),
                                         swapRB=False, crop=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]

            genders_detected.append(gender)

            # Display the gender label on the screen
            label = f"{gender}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return genders_detected, face_locations

    def analyze_group_and_trigger_alert(self, genders_detected, frame):
        """Analyze the group composition and trigger an alert if necessary."""
        men_count = genders_detected.count('Male')
        women_count = genders_detected.count('Female')

        # Determine context
        num_people = len(genders_detected)
        if num_people < 2:
            context = "Low Context Awareness"  # Not enough people to assess context
        else:
            context = "High Context Awareness"  # Sufficient number of people for context analysis

        # If the group consists of mostly men and at least one woman, consider it suspicious
        if men_count >= 2 and women_count == 1:
            self.trigger_alert_and_save_image(frame, "Hand Raised Alert by Woman", context, genders_detected.count('Male'), genders_detected.count('Female'))
    def trigger_alert_and_save_image(self, frame, alert_type, context, men_count, women_count):
        """Simulate alert and capture image when suspicious situation is detected, with context information."""
        # Get the current time, date, and GPS location
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        location = self.get_current_location()
        location_str = f"{location[0]}, {location[1]}" if location else "Location not available"

        # Save the image with a timestamp, location, and context information
        file_name = f"{alert_type}_{current_time}.png"
        file_path = os.path.join(self.save_directory, file_name)
        cv2.imwrite(file_path, frame)

        print(f"ALERT: {alert_type}! Image saved as {file_path}")
        print(f"Time: {current_time}, Location: {location_str}, Context: {context}")

        # Save details in a text file
        with open(os.path.join(self.save_directory, "incident_log.txt"), "a") as log_file:
            log_file.write(
                f"ALERT: {alert_type}\nTime: {current_time}\nLocation: {location_str}\nContext: {context}\n"
                f"Men Count: {men_count}, Women Count: {women_count}\nImage: {file_name}\n\n"
            )

    def run(self):
        """Main method to run gesture and group analysis."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB for Mediapipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe to detect hands
            frame_count = 0
            if frame_count % 5 == 0:  # Adjust the frequency of analysis
                results = self.hands.process(frame_rgb)

            # Detect faces and classify genders
            genders_detected, _ = self.detect_faces_and_gender(frame)

            # If hands are detected, perform gesture recognition only if a woman is detected in the frame
            if results.multi_hand_landmarks:
                self.detect_gestures(frame, results, genders_detected)

            # Analyze the group and trigger alert if necessary
            self.analyze_group_and_trigger_alert(genders_detected, frame)

            # Show the frame
            cv2.imshow('Safety Alert System', frame)

            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()


# Create an instance of the class and run the program
if __name__ == "__main__":
    system = SafetyAlertSystem()
    system.run()