from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2 as cv
import numpy as np
import insightface
import faiss
import pickle
import time
import threading
from queue import Queue
import base64
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# Initialize InsightFace model
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

# Load embeddings and labels
with open('labels.pkl', 'rb') as f:
    data = pickle.load(f)

Y = data['labels']

# Setup FAISS IndexFlatIP
index = faiss.read_index('face_index.faiss')

# MongoDB connection
mongo_client = MongoClient("mongodb+srv://sureshelite07:6NtP2zHJyxUJGrWy@cluster0.sc6f8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = mongo_client["face_recognition_db"]
known_faces_collection = db["known_faces"]
unknown_faces_collection = db["unknown_faces"]
stats_collection = db["detection_stats"]

# Clear all existing data in collections when starting the application
known_faces_collection.delete_many({})
unknown_faces_collection.delete_many({})
stats_collection.delete_many({})

# Initialize stats document
stats_collection.insert_one({
    "_id": "face_counter", 
    "known_count": 0,
    "unknown_count": 0
})

# Function to store face data in MongoDB
def store_face_in_db(face_data, is_known):
    face_id = face_data["face_id"]
    collection = known_faces_collection if is_known else unknown_faces_collection
    
    # Check if the face already exists in MongoDB (using face_id)
    existing_face = collection.find_one({"face_id": face_id})
    
    if not existing_face:
        # Insert the new face
        collection.insert_one(face_data)
        
        # Update the appropriate counter
        if is_known:
            stats_collection.update_one(
                {"_id": "face_counter"},
                {"$inc": {"known_count": 1}}
            )
        else:
            stats_collection.update_one(
                {"_id": "face_counter"},
                {"$inc": {"unknown_count": 1}}
            )

# Camera URLs
camera_urls = [
    0
]

# Desired resolution for processing/display
desired_width = 640
desired_height = 480

# Global set to track faces that have already been sent to frontend
sent_faces = set()

# Global dictionaries to store all detected faces for persistent display
all_known_faces = {}
all_unknown_faces = {}

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
        'running': True
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
                        is_known = matched_name != "Unknown"
                        color = (0, 255, 0) if is_known else (0, 0, 255)

                        # Encode the face image to base64
                        face_image = resized_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        _, buffer = cv.imencode('.jpg', face_image)
                        face_image_base64 = base64.b64encode(buffer).decode('utf-8')

                        # For "Unknown" faces, create a unique identifier based on face image
                        face_id = matched_name if is_known else f"Unknown_{hash(face_image_base64) % 10000000}"

                        face_data = {
                            'bbox': bbox,
                            'distance': distance[0],
                            'name': matched_name,
                            'color': color,
                            'face_image': face_image_base64,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'face_id': face_id
                        }
                        
                        result['faces'].append(face_data)
                        
                        # Create MongoDB document
                        face_db_entry = {
                            'face_id': face_id,
                            '_id': str(hash(face_image_base64)),
                            'name': matched_name,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'camera_id': camera_id,
                            'face_image': face_image_base64
                        }
                        
                        # Add face to the appropriate global dictionary for persistent display
                        if is_known:
                            all_known_faces[face_id] = face_db_entry.copy()
                        else:
                            all_unknown_faces[face_id] = face_db_entry.copy()
                        
                        # Store face in MongoDB
                        store_face_in_db(face_db_entry, is_known)

            result_queue.put(result)
            data['frame_queue'].task_done()

        except Exception as e:
            print(f"Camera {camera_id} Processing error: {e}")

# Initialize and start threads
threads = []
for i in camera_data:
    thread = threading.Thread(target=process_frames, args=(i,), daemon=True)
    threads.append(thread)
    camera_data[i]['running'] = True
    thread.start()

# Video capture setup
for i, url in enumerate(camera_urls):
    data = camera_data[i]
    data['video'] = cv.VideoCapture(url)

# Helper function to draw labels on the frame
def draw_label(img, text, pos, color, scale=0.7, thickness=2):
    cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness * 3)
    cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def generate_frames(camera_id):
    while camera_data[camera_id]['running']:
        ret, frame = camera_data[camera_id]['video'].read()
        if not ret:
            break

        resized_frame = cv.resize(frame, (desired_width, desired_height))

        current_time = time.time()

        camera_data[camera_id]['frame_times'].append(current_time)
        camera_data[camera_id]['frame_times'] = [t for t in camera_data[camera_id]['frame_times'] if t > current_time - 1]
        camera_data[camera_id]['display_fps'] = len(camera_data[camera_id]['frame_times'])

        if not camera_data[camera_id]['frame_queue'].full():
            camera_data[camera_id]['frame_queue'].put(frame.copy())

        while not camera_data[camera_id]['result_queue'].empty():
            camera_data[camera_id]['last_result'] = camera_data[camera_id]['result_queue'].get()
            camera_data[camera_id]['last_result_time'] = current_time

        if camera_data[camera_id]['last_result'] and current_time - camera_data[camera_id]['last_result_time'] < 0.5:
            camera_data[camera_id]['processing_times'].append(camera_data[camera_id]['last_result']['frame_time'])
            camera_data[camera_id]['processing_times'] = [t for t in camera_data[camera_id]['processing_times'] if t > current_time - 1]
            camera_data[camera_id]['processing_fps'] = len(camera_data[camera_id]['processing_times'])

            for face in camera_data[camera_id]['last_result']['faces']:
                bbox = face['bbox']
                name = face['name']
                distance = face['distance']
                color = face['color']

                cv.rectangle(resized_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                label = f"{name} ({distance:.2f})"
                draw_label(resized_frame, label, (bbox[0], bbox[1] - 10), color)

        # Get current face count from MongoDB
        counter_doc = stats_collection.find_one({"_id": "face_counter"})
        known_count = counter_doc.get("known_count", 0) if counter_doc else 0
        unknown_count = counter_doc.get("unknown_count", 0) if counter_doc else 0
        
        fps_text = f"Cam {camera_id} Display FPS: {camera_data[camera_id]['display_fps']} | Processing FPS: {camera_data[camera_id]['processing_fps']}"
        draw_label(resized_frame, fps_text, (10, 30), (0, 255, 0))
        
        # Display face counts
        known_text = f"Known Faces: {known_count}"
        unknown_text = f"Unknown Faces: {unknown_count}"
        draw_label(resized_frame, known_text, (10, 60), (0, 255, 0))
        draw_label(resized_frame, unknown_text, (10, 90), (0, 0, 255))

        ret, buffer = cv.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(0),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_data')
def detection_data():
    # Combine known and unknown faces but mark them accordingly
    all_faces = []
    
    for face_id, face_data in all_known_faces.items():
        face_data['is_known'] = True
        all_faces.append(face_data)
        if face_id not in sent_faces:
            sent_faces.add(face_id)
    
    for face_id, face_data in all_unknown_faces.items():
        face_data['is_known'] = False
        all_faces.append(face_data)
        if face_id not in sent_faces:
            sent_faces.add(face_id)
    
    return jsonify(all_faces)

@app.route('/known_faces')
def known_faces():
    known_faces_list = list(all_known_faces.values())
    return jsonify(known_faces_list)

@app.route('/unknown_faces')
def unknown_faces():
    unknown_faces_list = list(all_unknown_faces.values())
    return jsonify(unknown_faces_list)

@app.route('/face_count')
def face_count():
    # Get the current counts from MongoDB
    counter_doc = stats_collection.find_one({"_id": "face_counter"})
    
    if counter_doc:
        known_count = counter_doc.get("known_count", 0)
        unknown_count = counter_doc.get("unknown_count", 0)
    else:
        known_count = 0
        unknown_count = 0
    
    return jsonify({
        "known_count": known_count,
        "unknown_count": unknown_count,
        "total_count": known_count + unknown_count
    })

@app.route('/reset_faces', methods=['POST'])
def reset_faces():
    global sent_faces
    global all_known_faces
    global all_unknown_faces
    
    sent_faces.clear()
    all_known_faces.clear()
    all_unknown_faces.clear()
    
    # Reset MongoDB collections
    known_faces_collection.delete_many({})
    unknown_faces_collection.delete_many({})
    stats_collection.update_one(
        {"_id": "face_counter"},
        {"$set": {"known_count": 0, "unknown_count": 0}}
    )
    
    return jsonify({
        "status": "reset successful", 
        "known_count": 0,
        "unknown_count": 0,
        "message": "All detected faces have been cleared from memory and database"
    })

@app.route('/stop', methods=['POST'])
def stop():
    for camera_id in camera_data:
        camera_data[camera_id]['running'] = False
        if camera_data[camera_id]['video']:
            camera_data[camera_id]['video'].release()
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)