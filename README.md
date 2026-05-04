🚗 Drowsiness Detection System

A real-time driver drowsiness and distraction detection system built with Python, OpenCV, and MediaPipe.
The system continuously monitors a driver’s facial landmarks to detect early signs of fatigue or distraction and sends real-time alerts to a Firebase-powered dashboard.

✨ Features

👁️ Real-Time Drowsiness Detection: Calculates Eye Aspect Ratio (EAR) to detect closed or drooping eyes.

😮 Yawn Detection: Uses Mouth Aspect Ratio (MAR) to identify yawning patterns.

🧭 Distraction Detection: Estimates head pose to check if the driver is looking away from the road.

📊 Fatigue Scoring: Computes a cumulative fatigue index using weighted event tracking.

☁️ Firebase Integration: Sends real-time SOS alerts and GPS coordinates to a Flutter-based dashboard app.

🔊 Audible Alerts: Plays warning tones and voice prompts such as “Wake up” or “Pay attention.”

⚙️ Setup
1️⃣ Clone this repository
git clone <your-repository-url>

2️⃣ Install dependencies
pip install -r requirements.txt


(Ensure that requirements.txt lists all necessary libraries — OpenCV, Mediapipe, Firebase Admin SDK, etc.)

3️⃣ Configure Firebase

Create a new Firebase project in the Firebase Console
.

Go to Project Settings → Service Accounts.

Click “Generate new private key” and download the serviceAccountKey.json file.

Place it in the project’s root directory.
(This file is already ignored by .gitignore for security.)

▶️ Run the System

Ensure your webcam (or DroidCam stream) is active.
Then run:

python main.py


The program will start calibrating automatically and display live detection results.
Press ‘q’ to safely exit the system.

🧠 Tech Stack

Language: Python

Libraries: OpenCV, Mediapipe, NumPy, PyAudio, Firebase Admin SDK

Cloud Backend: Firebase Realtime Database

Front-End Dashboard: Flutter (companion app)

Deployment Target: Desktop / Laptop

🧩 Future Improvements

Integration of ML-based fatigue classification model.

Adaptive alert threshold based on driver profile.

Cloud synchronization of driver logs and performance.

🏷️ Authorship & License

Developed and maintained by Daksh Sharma
 (2025).
This project was originally designed, implemented, and published by Daksh Sharma as part of AI-driven road safety research and hackathon development work.

Licensed under the MIT License
.
Any derivative or adapted work must include proper credit to the original author.

⭐ Star this repo if you find it useful or want to follow its future AI-based fatigue detection upgrades.
