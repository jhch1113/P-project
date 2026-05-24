import cv2
import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
import time
import threading
from collections import deque
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
from contextlib import asynccontextmanager

# --- Global State Management ---
current_metrics = {"ear": 0.0, "mar": 0.0, "perclos": 0.0, "pitch": 0.0, "status": "Waiting..."}
output_ir = None
output_rgb = None
output_depth = None
lock = threading.Lock()

# --- DriverMonitorCV (Core Logic) ---
class DriverMonitorCV:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 13, 14, 81, 178, 311, 402]
        self.model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
        self.is_calibrated = False
        self.calibration_start_time = None
        self.calibration_duration = 5.0 
        self.ear_list = []              
        self.EAR_THRESHOLD = 0.25 
        self.MAR_THRESHOLD = 0.6   
        self.PITCH_THRESHOLD = 25  
        self.HEAD_DROP_FRAMES = 12 
        self.head_drop_counter = 0
        self.eye_closed_history = deque()       
        self.ear_buffer = deque(maxlen=5)       
        self.mar_buffer = deque(maxlen=5)       
        self.pitch_buffer = deque(maxlen=10)    
        self.perclos_window = 60                
        self.pitch_ema = None                   
        self.pitch_alpha = 0.7                  
        self.mar_ema = None                     
        self.mar_alpha = 0.6                    

    def get_ear(self, landmarks, indices):
        coords = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h_dist = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h_dist) if h_dist != 0 else 0

    def get_mar(self, landmarks, indices):
        def pt(idx): return np.array([landmarks[idx].x, landmarks[idx].y])
        horiz = np.linalg.norm(pt(61) - pt(291))
        verticals = [np.linalg.norm(pt(13) - pt(14)), np.linalg.norm(pt(81) - pt(178)), np.linalg.norm(pt(311) - pt(402))]
        vert = sum(verticals) / len(verticals)
        return vert / horiz if horiz != 0 else 0

    def get_head_pose(self, landmarks, w, h):
        image_points = np.array([(landmarks[1].x * w, landmarks[1].y * h), (landmarks[152].x * w, landmarks[152].y * h), (landmarks[263].x * w, landmarks[263].y * h), (landmarks[33].x * w, landmarks[33].y * h), (landmarks[291].x * w, landmarks[291].y * h), (landmarks[61].x * w, landmarks[61].y * h)], dtype="double")
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1)) 
        success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return None, None, None
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0] * 360, angles[1] * 360, angles[2] * 360 

    def process_frame(self, image_rgb):
        img_h, img_w, _ = image_rgb.shape
        results = self.face_mesh.process(image_rgb)
        data = {"ear": 0.0, "mar": 0.0, "perclos": 0.0, "pitch": 0.0, "status": "No Face"}

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            avg_ear = (self.get_ear(landmarks, self.LEFT_EYE) + self.get_ear(landmarks, self.RIGHT_EYE)) / 2.0
            self.ear_buffer.append(avg_ear)
            smoothed_ear = sum(self.ear_buffer) / len(self.ear_buffer)
            data["ear"] = smoothed_ear

            raw_mar = self.get_mar(landmarks, self.MOUTH)
            self.mar_buffer.append(raw_mar)
            smoothed_mar = sum(self.mar_buffer) / len(self.mar_buffer)
            if self.mar_ema is None: self.mar_ema = smoothed_mar
            else: self.mar_ema = self.mar_alpha * smoothed_mar + (1 - self.mar_alpha) * self.mar_ema
            data["mar"] = self.mar_ema

            pitch, yaw, roll = self.get_head_pose(landmarks, img_w, img_h)
            if pitch is not None:
                self.pitch_buffer.append(pitch)
                smoothed_pitch = sum(self.pitch_buffer) / len(self.pitch_buffer)
                if self.pitch_ema is None: self.pitch_ema = smoothed_pitch
                else: self.pitch_ema = self.pitch_alpha * smoothed_pitch + (1 - self.pitch_alpha) * self.pitch_ema
                data["pitch"] = self.pitch_ema

            current_time = time.time()
            if not self.is_calibrated:
                if self.calibration_start_time is None: self.calibration_start_time = current_time
                elapsed = current_time - self.calibration_start_time
                self.ear_list.append(smoothed_ear)
                data["status"] = f"Calibrating... {5 - int(elapsed)}s"
                if elapsed > self.calibration_duration:
                    self.EAR_THRESHOLD = (sum(self.ear_list) / len(self.ear_list)) * 0.8 
                    self.is_calibrated = True
            else:
                is_closed = smoothed_ear < self.EAR_THRESHOLD
                self.eye_closed_history.append((current_time, is_closed))
                while self.eye_closed_history and (current_time - self.eye_closed_history[0][0] > self.perclos_window):
                    self.eye_closed_history.popleft()
                if self.eye_closed_history:
                    closed_count = sum(1 for _, closed in self.eye_closed_history if closed)
                    data["perclos"] = (closed_count / len(self.eye_closed_history)) * 100

                warnings = []
                if is_closed: warnings.append("EYES CLOSED")
                if data["mar"] > self.MAR_THRESHOLD: warnings.append("YAWNING")
                if self.pitch_ema > self.PITCH_THRESHOLD: self.head_drop_counter += 1
                else: self.head_drop_counter = 0
                if self.head_drop_counter >= self.HEAD_DROP_FRAMES: warnings.append("HEAD DROP")
                data["status"] = ", ".join(warnings) if warnings else "NORMAL"
        return data, results.multi_face_landmarks

# --- Camera Streaming Background Loop (3 Active Streams) ---
def vision_processing_loop():
    global current_metrics, output_ir, output_rgb, output_depth
    detector = DriverMonitorCV()
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable all three streams
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    colorizer = rs.colorizer()  # Depth colorization tool
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    try:
        while True:
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not ir_frame or not color_frame or not depth_frame: continue

            # Convert frames to numpy arrays
            ir_img = np.asanyarray(ir_frame.get_data())
            rgb_img = np.asanyarray(color_frame.get_data())
            depth_color_img = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # Apply landmark analysis and overlay to the IR image
            ir_rgb = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)
            data, landmarks = detector.process_frame(ir_rgb)

            if landmarks:
                mp_drawing.draw_landmarks(ir_rgb, landmarks[0], mp_face_mesh.FACEMESH_TESSELATION, None, mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

            color = (0, 255, 0) if data["status"] == "NORMAL" or "Calibrating" in data["status"] else (0, 0, 255)
            cv2.putText(ir_rgb, f"State: {data['status']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(ir_rgb, f"EAR: {data['ear']:.3f} (Thresh: {detector.EAR_THRESHOLD:.3f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(ir_rgb, f"PERCLOS: {data['perclos']:.1f}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(ir_rgb, f"Head Pitch: {data['pitch']:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # JPEG-encode each of the three video streams (compute-intensive step)
            ret1, buf_ir = cv2.imencode('.jpg', ir_rgb)
            ret2, buf_rgb = cv2.imencode('.jpg', rgb_img)
            ret3, buf_depth = cv2.imencode('.jpg', depth_color_img)

            with lock:
                if ret1: output_ir = buf_ir.tobytes()
                if ret2: output_rgb = buf_rgb.tobytes()
                if ret3: output_depth = buf_depth.tobytes()
                current_metrics = data

    except Exception as e:
        print(f"Loop Error: {e}")
    finally:
        pipeline.stop()

# --- FastAPI Router and Endpoints ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=vision_processing_loop, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

def generate_video(stream_type):
    while True:
        with lock:
            if stream_type == 'ir': frame = output_ir
            elif stream_type == 'rgb': frame = output_rgb
            elif stream_type == 'depth': frame = output_depth
            else: frame = None
        
        if frame is None:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video/ir")
def video_ir(): return StreamingResponse(generate_video('ir'), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video/rgb")
def video_rgb(): return StreamingResponse(generate_video('rgb'), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video/depth")
def video_depth(): return StreamingResponse(generate_video('depth'), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/metrics")
def get_metrics():
    with lock: return current_metrics

@app.get("/")
def index():
    html_content = """
    <html>
        <head><title>Multi-Sensor Drowsiness Monitor</title></head>
        <body style="background-color: #121212; color: white; text-align: center; font-family: sans-serif;">
            <h1>RealSense Multi-Stream Dashboard</h1>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div><h3>IR (AI Analysis)</h3><img src="/video/ir" width="400"></div>
                <div><h3>RGB (Standard)</h3><img src="/video/rgb" width="400"></div>
                <div><h3>Depth (Colorized)</h3><img src="/video/depth" width="400"></div>
            </div>
            <div style="margin-top: 30px; background-color: #2d2d2d; padding: 20px; border-radius: 10px; display: inline-block; text-align: left;">
                <h2>Live Metrics</h2>
                <p><strong>EAR:</strong> <span id="ear">0.0</span></p>
                <p><strong>MAR:</strong> <span id="mar">0.0</span></p>
                <p><strong>PERCLOS:</strong> <span id="perclos">0.0</span> %</p>
                <h3 id="status" style="color: #4CAF50;">Status: AWAKE</h3>
            </div>
            <script>
                setInterval(() => {
                    fetch('/metrics')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('ear').innerText = data.ear.toFixed(3);
                            document.getElementById('mar').innerText = data.mar.toFixed(3);
                            document.getElementById('perclos').innerText = data.perclos.toFixed(1);
                            const statusElem = document.getElementById('status');
                            if(data.status !== "NORMAL" && !data.status.includes("Calibrating")) {
                                statusElem.innerText = "Status: " + data.status;
                                statusElem.style.color = "#ff4d4d";
                            } else {
                                statusElem.innerText = "Status: " + data.status;
                                statusElem.style.color = "#4CAF50";
                            }
                        });
                }, 100);
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run("multi_sensor_api:app", host="0.0.0.0", port=8000, reload=False)
