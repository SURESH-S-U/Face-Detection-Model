import cv2 as cv
import numpy as np
import insightface
import faiss
import pickle
import time
import threading
from queue import Queue
import base64
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import json
from db import FaceDBLoggerSQL

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize InsightFace model
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

# Initialize db
database = FaceDBLoggerSQL(host="127.0.0.1", port=3306)

# Load embeddings and labels
with open('labels.pkl', 'rb') as f:
    data = pickle.load(f)

Y = data['labels']

# Setup FAISS IndexFlatIP
index = faiss.read_index('face_index.faiss')

# Camera URLs
camera_urls = [0]  # Default to first camera

# Desired resolution for processing
desired_width = 640
desired_height = 480

# Data structures for each camera
camera_data = {}
for i, url in enumerate(camera_urls):
    camera_data[i] = {
        'frame_queue': Queue(maxsize=2),
        'result_queue': Queue(maxsize=2),
        'last_result': None,
        'last_result_time': 0,
        'display_fps': 0,
        'processing_fps': 0,
        'frame_times': [],
        'processing_times': [],
        'video': None,
        'running': True,
        'latest_frame': None,
        'latest_frame_with_annotations': None
    }

# Face processing thread
def process_frames(camera_id):
    data = camera_data[camera_id]
    result_queue = data['result_queue']
    while data['running']:
        try:
            frame = data['frame_queue'].get(timeout=0.3)

            # Resize frame to desired resolution for processing
            resized_frame = cv.resize(frame, (desired_width, desired_height))

            result = {
                'faces': [],
                'frame_time': time.time()
            }

            faces = model.get(resized_frame)

            if faces:
                embeddings = []
                face_data = []

                for face in faces:
                    bbox = face.bbox.astype(int)
                    embedding = face.embedding.astype(np.float32).reshape(1, -1)
                    embedding /= np.linalg.norm(embedding)

                    embeddings.append(embedding[0])
                    face_data.append((face, bbox))

                if embeddings:
                    embeddings = np.array(embeddings, dtype=np.float32)
                    distances, indices = index.search(embeddings, k=1)

                    for (face, bbox), distance, idx in zip(face_data, distances, indices):
                        matched_name = Y[idx[0]] if distance[0] > 0.5 else "Unknown"
                        color = (0, 255, 0) if matched_name != "Unknown" else (0, 0, 255)

                        result['faces'].append({
                            'bbox': bbox.tolist(),  # Convert to list for JSON serialization
                            'distance': float(distance[0]),  # Convert to float for JSON serialization
                            'name': matched_name,
                            'color': color
                        })
                        if distance[0] > 0.5 and database.should_log_detection(matched_name):
                            database.log_detection(matched_name, float(distance[0]), camera_id)

            result_queue.put(result)
            data['frame_queue'].task_done()

        except Exception as e:
            print(f"Camera {camera_id} Processing error: {e}")

# Helper function to draw annotations on frame
def draw_annotations(frame, result):
    # Make a copy to avoid modifying the original
    annotated_frame = frame.copy()
    
    for face in result['faces']:
        bbox = face['bbox']
        name = face['name']
        distance = face['distance']
        color = face['color']

        cv.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Draw label
        label = f"{name} ({distance:.2f})"
        cv.putText(annotated_frame, label, (bbox[0], bbox[1] - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 6)
        cv.putText(annotated_frame, label, (bbox[0], bbox[1] - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return annotated_frame

# Start camera thread function
def start_camera(camera_id, url):
    if camera_id in camera_data:
        camera_data[camera_id]['running'] = False
        time.sleep(1)  # Give time for thread to close
    
    # Setup new camera data
    camera_data[camera_id] = {
        'frame_queue': Queue(maxsize=2),
        'result_queue': Queue(maxsize=2),
        'last_result': None,
        'last_result_time': 0,
        'display_fps': 0,
        'processing_fps': 0,
        'frame_times': [],
        'processing_times': [],
        'video': cv.VideoCapture(url),
        'running': True,
        'latest_frame': None,
        'latest_frame_with_annotations': None
    }
    
    # Start processing thread
    thread = threading.Thread(target=process_frames, args=(camera_id,), daemon=True)
    thread.start()
    
    return thread

# Initialize and start camera threads
camera_threads = {}
for i, url in enumerate(camera_urls):
    camera_threads[i] = start_camera(i, url)

# Function to capture and process frames continuously
def capture_frames():
    while True:
        current_time = time.time()
        
        for i in camera_data:
            data = camera_data[i]
            if not data['running']:
                continue
                
            ret, frame = data['video'].read()
            if not ret:
                print(f"Failed to read from camera {i}")
                continue
                
            # Resize frame for processing
            resized_frame = cv.resize(frame, (desired_width, desired_height))
            
            # Store the latest frame
            _, buffer = cv.imencode('.jpg', resized_frame)
            data['latest_frame'] = buffer.tobytes()
            
            # Update FPS tracking
            data['frame_times'].append(current_time)
            data['frame_times'] = [t for t in data['frame_times'] if t > current_time - 1]
            data['display_fps'] = len(data['frame_times'])
            
            # Queue frame for processing
            if not data['frame_queue'].full():
                data['frame_queue'].put(resized_frame.copy())
                
            # Retrieve latest result
            while not data['result_queue'].empty():
                data['last_result'] = data['result_queue'].get()
                data['last_result_time'] = current_time
                
                # Update processing FPS
                data['processing_times'].append(data['last_result']['frame_time'])
                data['processing_times'] = [t for t in data['processing_times'] if t > current_time - 1]
                data['processing_fps'] = len(data['processing_times'])
                
                # Create annotated frame
                if data['last_result'] and data['latest_frame']:
                    decoded_frame = cv.imdecode(np.frombuffer(data['latest_frame'], np.uint8), cv.IMREAD_COLOR)
                    annotated_frame = draw_annotations(decoded_frame, data['last_result'])
                    _, buffer = cv.imencode('.jpg', annotated_frame)
                    data['latest_frame_with_annotations'] = buffer.tobytes()
        
        time.sleep(0.01)  # Small delay to prevent high CPU usage

# Start the frame capture in a separate thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# API endpoints
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    camera_list = []
    for i in camera_data:
        camera_list.append({
            'id': i,
            'name': f'Camera {i}',
            'status': 'active' if camera_data[i]['running'] else 'inactive',
            'fps': camera_data[i]['processing_fps']
        })
    return jsonify(camera_list)

@app.route('/api/frame/<int:camera_id>', methods=['GET'])
def get_frame(camera_id):
    if camera_id not in camera_data:
        return jsonify({'error': 'Camera not found'}), 404
        
    data = camera_data[camera_id]
    
    if data['latest_frame_with_annotations']:
        img_base64 = base64.b64encode(data['latest_frame_with_annotations']).decode('utf-8')
        return jsonify({
            'frame': img_base64,
            'display_fps': data['display_fps'],
            'processing_fps': data['processing_fps']
        })
    else:
        return jsonify({'error': 'No frame available'}), 404

@app.route('/api/results/<int:camera_id>', methods=['GET'])
def get_results(camera_id):
    if camera_id not in camera_data:
        return jsonify({'error': 'Camera not found'}), 404
        
    data = camera_data[camera_id]
    
    if data['last_result'] and time.time() - data['last_result_time'] < 5:
        return jsonify({
            'faces': data['last_result']['faces'],
            'timestamp': data['last_result_time'],
            'processing_fps': data['processing_fps']
        })
    else:
        return jsonify({'faces': [], 'timestamp': time.time(), 'processing_fps': data['processing_fps']})

@app.route('/api/recognized_users', methods=['GET'])
def get_recognized_users():
    # Get the recent known and unknown users from the database
    known_users = []
    unknown_users = []
    
    for i in camera_data:
        data = camera_data[i]
        if data['last_result'] and time.time() - data['last_result_time'] < 30:
            for face in data['last_result']['faces']:
                if face['name'] != "Unknown":
                    # Get user image from the frame for known users
                    known_users.append({
                        'id': len(known_users) + 1,
                        'name': face['name'],
                        'time': time.strftime('%I:%M %p', time.localtime(data['last_result_time'])),
                        'camera': f'Camera {i}',
                        'confidence': face['distance']
                    })
                else:
                    # Unknown user
                    unknown_users.append({
                        'id': len(unknown_users) + 1,
                        'time': time.strftime('%I:%M %p', time.localtime(data['last_result_time'])),
                        'camera': f'Camera {i}'
                    })
    
    # Remove duplicates by name (keep most recent for each person)
    known_dict = {}
    for user in known_users:
        known_dict[user['name']] = user
    
    return jsonify({
        'known_users': list(known_dict.values()),
        'unknown_users': unknown_users[:10]  # Limit to 10 unknown users
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)