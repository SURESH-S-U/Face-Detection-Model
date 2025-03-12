from flask import Flask, Response
import cv2
import threading
import time
import numpy as np
import insightface
import faiss
import pickle
from datetime import datetime
import pymongo
from pymongo import MongoClient
import gridfs
import ssl
from queue import Queue

# Initialize Flask app for streaming
app = Flask(__name__)

# Global variable to store the latest frame
latest_frame = None
frame_lock = threading.Lock()

def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask_app():
    app.run(host='0.0.0.0', port=5000, threaded=True)

# Start Flask app in a separate thread
flask_thread = threading.Thread(target=start_flask_app)
flask_thread.daemon = True
flask_thread.start()

# Rest of your existing code (face detection logic)
def get_mongodb_connection():
    try:
        connection_string = "mongodb+srv://sureshelite07:xnwMuZHq04ipgHpm@cluster0.gsjwi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tlsInsecure=true"
        client = MongoClient(connection_string)
        print("MongoDB connection successful!")
        db = client["face_detection_db"]
        return db
    except Exception as e:
        print(f"MongoDB Connection Error: {e}")
        return None

# FaceDB Logger for MongoDB
class FaceDBLoggerMongoDB:
    def __init__(self):
        self.db = get_mongodb_connection()
        if self.db is None:
            print("WARNING: MongoDB connection failed, detection logging disabled")
            return
            
        self.detections_collection = self.db["face_detections"]
        self.stats_collection = self.db["face_stats"]
        self.fs = gridfs.GridFS(self.db)  # Initialize GridFS for storing images
        
        # Clear existing data in the collections
        self.clear_existing_data()
        
        # Set to track logged persons
        self.logged_persons = set()
    
    def clear_existing_data(self):
        """Clear existing data in the face_detections and face_stats collections"""
        try:
            # Delete all documents in the face_detections collection
            self.detections_collection.delete_many({})
            print("Cleared existing data in face_detections collection.")
            
            # Delete all documents in the face_stats collection
            self.stats_collection.delete_many({})
            print("Cleared existing data in face_stats collection.")
            
            # Delete all files in GridFS (if any)
            for grid_file in self.fs.find():
                self.fs.delete(grid_file._id)
            print("Cleared existing files in GridFS.")
        except Exception as e:
            print(f"Error clearing existing data: {e}")
    
    def log_detection(self, name, confidence, camera_id, face_image=None):
        """Log a face detection to MongoDB"""
        if self.db is None:
            return
            
        try:
            # Skip logging if the person has already been logged
            if name in self.logged_persons:
                print(f"Person {name} already logged. Skipping...")
                return
            
            detection_time = datetime.now()
            detection_doc = {
                "name": name,
                "roll_number": self._extract_roll_number(name),
                "confidence": confidence,
                "camera_id": camera_id,
                "timestamp": detection_time
            }
            
            # Store the face image in GridFS if provided
            if face_image is not None:
                face_image_id = self.fs.put(face_image.tobytes(), filename=f"{name}_{detection_time}.jpg")
                detection_doc["face_image_id"] = face_image_id
            
            self.detections_collection.insert_one(detection_doc)
            print(f"Logged detection: {name} with confidence {confidence:.2f}")
            
            # Add the person to the logged_persons set
            self.logged_persons.add(name)
            
            # Update stats document for this person
            self._update_stats(name, detection_time)
        except Exception as e:
            print(f"Error logging detection: {e}")
    
    def _extract_roll_number(self, name):
        """Extract roll number from name if available"""
        return name
    
    def _update_stats(self, name, detection_time):
        """Update statistics for this person"""
        try:
            stats_query = {"name": name}
            stats_doc = self.stats_collection.find_one(stats_query)
            
            if stats_doc:
                # Update existing stats
                self.stats_collection.update_one(
                    stats_query,
                    {
                        "$inc": {"total_detections": 1},
                        "$set": {"last_detection_time": detection_time}
                    }
                )
            else:
                # Create new stats document
                new_stats = {
                    "name": name,
                    "roll_number": self._extract_roll_number(name),
                    "total_detections": 1,
                    "first_detection_time": detection_time,
                    "last_detection_time": detection_time,
                    "dataset_count": 25  # This would need to be updated manually
                }
                self.stats_collection.insert_one(new_stats)
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def get_detections_in_timerange(self, start_time, end_time):
        """Get count of detections within a time range"""
        if self.db is None:
            return 0
            
        query = {
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }
        return self.detections_collection.count_documents(query)
    
    def get_detections_after_time(self, time_point):
        """Get count of detections after a specific time"""
        if self.db is None:
            return 0
            
        query = {
            "timestamp": {
                "$gt": time_point
            }
        }
        return self.detections_collection.count_documents(query)
    
    def update_dataset_count(self, name, count):
        """Update the count of datasets provided for this person"""
        if self.db is None:
            return
            
        stats_query = {"name": name}
        self.stats_collection.update_one(
            stats_query,
            {"$set": {"dataset_count": count}},
            upsert=True
        )

# Initialize InsightFace model with explicit GPU context
try:
    model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    model.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace model initialized successfully")
except Exception as e:
    print(f"Error initializing InsightFace model: {e}")
    model = None
    exit(1)

# Initialize MongoDB database
database = FaceDBLoggerMongoDB()

# Load embeddings and labels
try:
    with open('labels.pkl', 'rb') as f:
        data = pickle.load(f)
    
    Y = data['labels']
    print(f"Loaded {len(Y)} face labels")
except Exception as e:
    print(f"Error loading face labels: {e}")
    Y = []
    exit(1)

# Setup FAISS IndexFlatIP
try:
    index = faiss.read_index('face_index.faiss')
    print(f"FAISS index loaded with {index.ntotal} vectors")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    index = None
    exit(1)

# Camera URLs
camera_urls = [
    "http://192.168.137.208:4747/video"  # Default webcam
]

# Desired resolution for processing/display
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
        'running': True
    }

# Face processing thread
def process_frames(camera_id):
    data = camera_data[camera_id]
    result_queue = data['result_queue']  # Get the camera-specific result queue
    while data['running']:
        try:
            frame = data['frame_queue'].get(timeout=0.3)

            # Resize frame to desired resolution for processing
            resized_frame = cv2.resize(frame, (desired_width, desired_height))

            result = {
                'faces': [],
                'frame_time': time.time()
            }

            # Get face detections from the model
            faces = model.get(resized_frame)  # Use resized_frame for face detection
            
            if faces:
                print(f"Detected {len(faces)} faces in camera {camera_id}")
                embeddings = []
                face_data = []

                for face in faces:
                    bbox = face.bbox.astype(int)
                    embedding = face.embedding
                    
                    if embedding is None:
                        continue
                        
                    embedding = embedding.astype(np.float32).reshape(1, -1)
                    embedding /= np.linalg.norm(embedding)  # Ensure normalized (cosine/IP fix)

                    embeddings.append(embedding[0])
                    face_data.append((face, bbox))

                if embeddings:
                    embeddings = np.array(embeddings, dtype=np.float32)
                    distances, indices = index.search(embeddings, k=1)

                    for i, ((face, bbox), distance, idx) in enumerate(zip(face_data, distances, indices)):
                        # Use a confidence threshold that makes sense for your model
                        confidence_threshold = 0.5
                        
                        # The distance might need to be converted to a similarity score
                        # For inner product similarity, higher is better
                        similarity = float(distance[0])
                        
                        matched_name = Y[idx[0]] if similarity > confidence_threshold else "Unknown"
                        color = (0, 255, 0) if matched_name != "Unknown" else (0, 0, 255)

                        print(f"Face {i+1}: Name={matched_name}, Similarity={similarity:.2f}")

                        result['faces'].append({
                            'bbox': bbox,
                            'distance': similarity,
                            'name': matched_name,
                            'color': color
                        })
                        
                        # Log the detection if confidence is high enough and not Unknown
                        if similarity > confidence_threshold and matched_name != "Unknown":
                            # Extract the face image from the frame
                            face_image = resized_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                            
                            print(f"Logging detection for {matched_name}")
                            database.log_detection(matched_name, similarity, camera_id, face_image)

            result_queue.put(result)
            data['frame_queue'].task_done()

        except Exception as e:
            print(f"Camera {camera_id} Processing error: {e}")
            import traceback
            traceback.print_exc()

# Initialize and start threads
threads = []
for i in camera_data:
    thread = threading.Thread(target=process_frames, args=(i,), daemon=True)
    threads.append(thread)
    camera_data[i]['running'] = True  # Ensure the running flag is set before starting
    thread.start()

# Video capture setup
for i, url in enumerate(camera_urls):
    data = camera_data[i]
    data['video'] = cv2.VideoCapture(url)
    if not data['video'].isOpened():
        print(f"Failed to open camera {i} with URL {url}")
    else:
        print(f"Successfully opened camera {i}")

# Helper function to draw labels on the frame
def draw_label(img, text, pos, color, scale=0.7, thickness=2):
    """Improved text rendering for readability."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness * 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

print("Starting main loop...")

while True:
    frames = []
    for i in camera_data:
        data = camera_data[i]
        ret, frame = data['video'].read()
        if not ret:
            print(f"Failed to read frame from camera {i}")
            data['running'] = False
            continue

        # Resize frame to desired resolution for display
        resized_frame = cv2.resize(frame, (desired_width, desired_height))

        current_time = time.time()

        # FPS - Display
        data['frame_times'].append(current_time)
        data['frame_times'] = [t for t in data['frame_times'] if t > current_time - 1]
        data['display_fps'] = len(data['frame_times'])

        if not data['frame_queue'].full():
            data['frame_queue'].put(frame.copy())  # Put the original frame, not resized

        # Retrieve latest result if available
        while not data['result_queue'].empty():
            data['last_result'] = data['result_queue'].get()
            data['last_result_time'] = current_time

        # Draw latest face results (if recent)
        if data['last_result'] and current_time - data['last_result_time'] < 0.5:
            data['processing_times'].append(data['last_result']['frame_time'])
            data['processing_times'] = [t for t in data['processing_times'] if t > current_time - 1]
            data['processing_fps'] = len(data['processing_times'])

            for face in data['last_result']['faces']:
                bbox = face['bbox']
                name = face['name']
                distance = face['distance']
                color = face['color']

                cv2.rectangle(resized_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                label = f"{name} ({distance:.2f})"
                draw_label(resized_frame, label, (bbox[0], bbox[1] - 10), color)

        # Display FPS overlay
        fps_text = f"Cam {i} Display FPS: {data['display_fps']} | Processing FPS: {data['processing_fps']}"
        draw_label(resized_frame, fps_text, (10, 30), (0, 255, 0))

        # Update the latest_frame for streaming
        with frame_lock:
            latest_frame = resized_frame

        frames.append(resized_frame)

    # Concatenate frames horizontally to display side by side
    if frames:
        if len(frames) == 4:
            c1 = np.hstack((frames[0], frames[1]))
            c2 = np.hstack((frames[2], frames[3]))
            combined_frame = np.vstack((c1, c2))
        else:
            combined_frame = np.hstack(frames)

        # Show the combined frame with all cameras' outputs side by side
        cv2.imshow('Combined Camera Feed', combined_frame)

    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

# Clean up
for i in camera_data:
    camera_data[i]['running'] = False

for thread in threads:
    thread.join(timeout=1.0)

for i in camera_data:
    data = camera_data[i]
    if data['video']:
        data['video'].release()

cv2.destroyAllWindows()