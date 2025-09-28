import cv2
import mediapipe as mp
import time
import math
import os
from datetime import datetime
import winsound
import threading
import firebase_admin
from firebase_admin import credentials, firestore
import pyttsx3
import whereami
import numpy as np

class DrowsinessDetector:
    def __init__(self):
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate("serviceAccountKey.json")
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            print("✅ Firebase initialized successfully.")
        except Exception as e:
            print(f"🔥 Firebase initialization failed: {e}")
            self.db = None
        self.ear_thresh = 0.25
        self.mar_thresh = 0.30
        self.closed_eye_duration = 0.8
        self.yawn_duration = 0.8
        self.distraction_duration_thresh = 2.0
        self.fatigue_level = 0
        self.warning_lvl = 4
        self.alert_lvl = 8
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [13, 14, 78, 308]
        self.NOSE = 1
        self.eye_closed_start = None
        self.yawn_start = None
        self.distraction_start = None
        self.last_decay = time.time()
        self.ss_dir = "screenshots_log"
        os.makedirs(self.ss_dir, exist_ok=True)
        self.last_ss_time = 0
        self.ss_cooldown = 5.0
        self.alarm_playing = False
        self.alarm_thread = None
        self.last_firebase_alert_time = 0
        self.firebase_alert_cooldown = 15
        self.tts_engine = pyttsx3.init()
        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

    def calibrate(self, cap):
        print("📢 Starting calibration... Please look straight at the camera with a neutral expression for 5 seconds.")
        ear_values = []
        mar_values = []
        start_time = time.time()
        while time.time() - start_time < 5.0:
            ret, frame = cap.read()
            if not ret:
                continue
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            cv2.putText(frame, "CALIBRATING...", (w // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Look straight with eyes open", (w // 2 - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Drowsiness Detection", frame)
            cv2.waitKey(1)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                ear = (self.get_ear(landmarks, self.LEFT_EYE) + self.get_ear(landmarks, self.RIGHT_EYE)) / 2.0
                mar = self.get_mar(landmarks)
                ear_values.append(ear)
                mar_values.append(mar)
        if not ear_values or not mar_values:
            print("🔥 Calibration failed: No face detected. Using default thresholds.")
            return
        avg_ear = sum(ear_values) / len(ear_values)
        avg_mar = sum(mar_values) / len(mar_values)
        self.ear_thresh = avg_ear * 0.85
        self.mar_thresh = avg_mar + 0.08
        print(f"✅ Calibration complete!")
        print(f"      -> New EAR Threshold: {self.ear_thresh:.2f}")
        print(f"      -> New MAR Threshold: {self.mar_thresh:.2f}")
        time.sleep(2)

    def _dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def get_ear(self, landmarks, eye_indices):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
        v1 = self._dist(p2, p6)
        v2 = self._dist(p3, p5)
        h = self._dist(p1, p4)
        return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

    def get_mar(self, landmarks):
        top, bottom, left, right = [landmarks[i] for i in self.MOUTH]
        v = self._dist(top, bottom)
        h = self._dist(left, right)
        return v / h if h != 0 else 0.0

    def get_head_pose(self, landmarks, frame_shape):
        h, w = frame_shape
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[152].x * w, landmarks[152].y * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[33].x * w, landmarks[33].y * h),
            (landmarks[287].x * w, landmarks[287].y * h),
            (landmarks[57].x * w, landmarks[57].y * h)
        ], dtype=np.float64)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        dist_coeffs = np.zeros((4, 1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs
        )
        (nose_end_point2D, _) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
        )
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        yaw = angles[1]
        pitch = angles[0]
        roll = angles[2]
        return yaw, pitch, roll, nose_end_point2D, image_points[0]

    def send_sos_alert(self):
        if self.db is None:
            print("🔥 Cannot send alert, Firebase not connected.")
            return
        now = time.time()
        if now - self.last_firebase_alert_time < self.firebase_alert_cooldown:
            return
        print("🚀 Sending SOS alert to Firebase...")
        print("🌍 Fetching current location...")
        try:
            location_tuple = whereami.whereami()
            latitude, longitude = location_tuple[0], location_tuple[1]
            print(f"✅ Location found: Lat={latitude}, Lng={longitude}")
        except Exception as e:
            print(f"⚠️ Could not fetch accurate location: {e}. Using default coordinates.")
            latitude, longitude = (28.9845, 77.7064) 
        try:
            alert_data = {
                'driverId': 'DRIVER_001',
                'timestamp': firestore.SERVER_TIMESTAMP,
                'location': {'latitude': latitude, 'longitude': longitude},
                'message': 'Drowsiness alert triggered!'
            }
            self.db.collection('alerts').add(alert_data)
            self.last_firebase_alert_time = now
            print("✅ Alert sent successfully.")
        except Exception as e:
            print(f"🔥 FAILED to send Firebase alert: {e}")

    def take_screenshot(self, frame, event_name):
        now = time.time()
        if now - self.last_ss_time > self.ss_cooldown:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(self.ss_dir, f"{event_name}_{ts}.jpg")
            cv2.imwrite(fn, frame)
            self.last_ss_time = now

    def _alarm_sound_function(self):
        self.alarm_playing = True
        for _ in range(3):
            winsound.Beep(1000, 300)
            time.sleep(0.1)
        self.alarm_playing = False

    def play_alarm(self):
        if not self.alarm_playing:
            self.alarm_thread = threading.Thread(target=self._alarm_sound_function)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()

    def _voice_alert_function(self, text):
        self.alarm_playing = True
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        self.alarm_playing = False

    def play_voice_alert(self, text):
        if not self.alarm_playing:
            self.alarm_thread = threading.Thread(target=self._voice_alert_function, args=(text,))
            self.alarm_thread.daemon = True
            self.alarm_thread.start()

    def analyze_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if time.time() - self.last_decay > 2.0 and self.fatigue_level > 0:
            self.fatigue_level -= 1
            self.last_decay = time.time()
        status = "AWAKE"
        color = (0, 255, 0)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            yaw, pitch, roll, nose_end, nose_start = self.get_head_pose(landmarks, (h, w))
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            is_attentive = not (yaw > 25 or yaw < -25 or pitch > 20 or pitch < -25)
            if is_attentive:
                self.distraction_start = None
                ear = (self.get_ear(landmarks, self.LEFT_EYE) + self.get_ear(landmarks, self.RIGHT_EYE)) / 2.0
                mar = self.get_mar(landmarks)
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if ear < self.ear_thresh:
                    if self.eye_closed_start is None: 
                        self.eye_closed_start = time.time()
                    elif time.time() - self.eye_closed_start > self.closed_eye_duration:
                        self.fatigue_level += 2
                        self.eye_closed_start = None
                else:
                    self.eye_closed_start = None
                if mar > self.mar_thresh:
                    if self.yawn_start is None: 
                        self.yawn_start = time.time()
                    elif time.time() - self.yawn_start > self.yawn_duration:
                        self.fatigue_level += 1
                        self.yawn_start = None
                else:
                    self.yawn_start = None
            else:
                if self.distraction_start is None:
                    self.distraction_start = time.time()
                elif time.time() - self.distraction_start > self.distraction_duration_thresh:
                    self.fatigue_level += 2
                    self.take_screenshot(frame, "distraction_alert")
                    status = "ALERT: PAY ATTENTION!"
                    color = (0, 0, 255)
                    self.play_voice_alert("Pay Attention")
                    self.distraction_start = time.time()
        cv2.putText(frame, f"Fatigue: {self.fatigue_level}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.fatigue_level >= self.alert_lvl:
            status = "ALERT! DROWSY!"
            color = (0, 0, 255)
            self.take_screenshot(frame, "fatigue_alert")
            self.play_voice_alert("Wake up")
            self.send_sos_alert()
        elif self.fatigue_level >= self.warning_lvl:
            status = "Warning: Drowsy"
            color = (0, 255, 255)
            self.play_alarm()
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status}", (w // 2 - 150, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open system camera.")
        return
    detector = DrowsinessDetector()
    detector.calibrate(cap)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        processed_frame = detector.analyze_frame(frame)
        cv2.imshow("Drowsiness Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
