import cv2 as cv
import numpy as np
import insightface
import faiss
import pickle
import time
import threading
import base64
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timedelta
import os
from pymongo import MongoClient
from bson.binary import Binary  # For storing binary data (images) in MongoDB
import json  # For pretty printing dictionaries

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize InsightFace model
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

# Initialize MongoDB client
mongo_client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB connection string
db = mongo_client["face_detection_db"]  # Database name
collection = db["detected_faces"]  # Collection name

# Clear existing MongoDB data when starting
collection.delete_many({})
print("="*70)
print("MongoDB data cleared on startup")
print("="*70)

# Load embeddings and labels
with open('labels.pkl', 'rb') as f:
    data = pickle.load(f)

Y = data['labels']

# Setup FAISS IndexFlatIP
index = faiss.read_index('face_index.faiss')

# Camera URLs
camera_urls = [0]

# Desired resolution for processing/display
desired_width = 640
desired_height = 480

# Threshold for face recognition confidence
FACE_RECOGNITION_THRESHOLD = 0.5

# Dictionary to track last detection times for deduplication
face_detection_times = {}  # Format: {person_name: last_detection_time}
DEDUPLICATION_WINDOW_MINUTES = 5  # 5-minute window for deduplication

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
        'original_width': 0,  # Will store the original frame width
        'original_height': 0,  # Will store the original frame height
        'processed_frame': None,  # Store the latest processed frame for streaming
        'processed_frame_lock': threading.Lock()  # Lock for thread-safe access to processed_frame
    }

# Pydantic model for API response
class DetectionResult(BaseModel):
    person_name: str
    camera_number: int
    date_time: str
    accuracy_percentage: float

# Function to encode image as binary data
def encode_image_to_binary(image):
    """Encode OpenCV image to MongoDB binary format"""
    _, buffer = cv.imencode('.jpg', image)  # Encode image as JPEG
    return Binary(buffer.tobytes())  # Convert to binary format

# Function to encode image to base64 for frontend
def encode_image_to_base64(image):
    """Encode OpenCV image to base64 string for web display"""
    _, buffer = cv.imencode('.jpg', image)  # Encode image as JPEG
    return base64.b64encode(buffer.tobytes()).decode('utf-8')  # Convert to base64 string

# Function to check if a face was recently detected (for deduplication)
def should_store_detection(person_name):
    """
    Determine if a detection should be stored in MongoDB based on time window.
    Returns True if the detection should be stored, False otherwise.
    """
    current_time = datetime.now()
    
    # Always store unknown faces
    if person_name == "Unknown":
        return True
        
    # Check if this person was detected recently
    if person_name in face_detection_times:
        last_detection = face_detection_times[person_name]
        time_difference = current_time - last_detection
        
        # If detection is within the deduplication window, don't store
        if time_difference < timedelta(minutes=DEDUPLICATION_WINDOW_MINUTES):
            print(f"Skipping duplicate detection for {person_name} (last seen {int(time_difference.total_seconds())} seconds ago)")
            return False
    
    # Update the last detection time for this person
    face_detection_times[person_name] = current_time
    return True

# Function to print MongoDB data (excluding binary image for readability)
def print_mongodb_data(data, source="camera"):
    """Print MongoDB data in readable format without the binary image"""
    # Create a copy of the data to avoid modifying the original
    printable_data = data.copy()
    
    # Remove binary image data for printing (too large to display)
    if 'face_image' in printable_data:
        printable_data['face_image'] = f"<Binary Data: {len(data['face_image'])} bytes>"
    
    # Format the output
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] MongoDB Entry from {source}:")
    print(f"  Person: {printable_data.get('person_name', 'Unknown')}")
    print(f"  Camera: {printable_data.get('camera_number', 'N/A')}")
    print(f"  Time: {printable_data.get('date_time', 'N/A')}")
    print(f"  Accuracy: {printable_data.get('accuracy_percentage', 0):.2f}%")
    if 'image_path' in printable_data:
        print(f"  Image: {printable_data['image_path']}")
    print(f"  Face Image: {printable_data.get('face_image', 'None')}")
    print("-" * 60)

# Face processing thread
def process_frames(camera_id):
    data = camera_data[camera_id]
    result_queue = data['result_queue']  # Get the camera-specific result queue
    while data['running']:
        try:
            frame = data['frame_queue'].get(timeout=0.3)
            
            # Store original dimensions
            orig_height, orig_width = frame.shape[:2]
            
            # Resize frame to desired resolution for processing
            resized_frame = cv.resize(frame, (desired_width, desired_height))

            result = {
                'faces': [],
                'frame_time': time.time(),
                'orig_width': orig_width,
                'orig_height': orig_height
            }

            faces = model.get(resized_frame)  # Use resized_frame for face detection

            if faces:
                embeddings = []
                face_data = []

                for face in faces:
                    bbox = face.bbox.astype(int)
                    embedding = face.embedding.astype(np.float32).reshape(1, -1)
                    embedding /= np.linalg.norm(embedding)  # Ensure normalized (cosine/IP fix)

                    embeddings.append(embedding[0])
                    face_data.append((face, bbox))

                if embeddings:
                    embeddings = np.array(embeddings, dtype=np.float32)
                    distances, indices = index.search(embeddings, k=1)

                    for (face, bbox), distance, idx in zip(face_data, distances, indices):
                        # Convert similarity score to confidence (0-1 range)
                        confidence = float(distance[0])
                        matched_name = Y[idx[0]] if confidence > FACE_RECOGNITION_THRESHOLD else "Unknown"
                        color = (0, 255, 0) if matched_name != "Unknown" else (0, 0, 255)

                        # Make sure bbox coordinates are within the image dimensions
                        bbox = [
                            max(0, min(bbox[0], orig_width-1)),
                            max(0, min(bbox[1], orig_height-1)),
                            max(0, min(bbox[2], orig_width-1)),
                            max(0, min(bbox[3], orig_height-1))
                        ]
                        
                        # Check if the bbox has valid dimensions
                        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                            continue  # Skip invalid bounding boxes
                            
                        # Crop the detected face from the original frame
                        try:
                            face_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            
                            # Check if face_image is empty
                            if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
                                continue  # Skip empty face images
                                
                            result['faces'].append({
                                'bbox': bbox,  # Store as list
                                'confidence': confidence,
                                'name': matched_name,
                                'color': color,
                                'face_image': face_image  # Store the cropped face image
                            })

                            # Log detection in MongoDB for recognized faces
                            if confidence > FACE_RECOGNITION_THRESHOLD:
                                # Check if this face should be stored (deduplication)
                                if should_store_detection(matched_name):
                                    # Create MongoDB document
                                    detection_data = {
                                        "person_name": matched_name,
                                        "camera_number": camera_id,
                                        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "accuracy_percentage": confidence * 100,
                                        "face_image": encode_image_to_binary(face_image)
                                    }
                                    
                                    # Print the data being sent to MongoDB
                                    print_mongodb_data(detection_data, "camera_feed")
                                    
                                    # Insert into MongoDB
                                    collection.insert_one(detection_data)
                            # Also store unknown faces in MongoDB
                            else:
                                # Create MongoDB document for unknown face
                                detection_data = {
                                    "person_name": "Unknown",
                                    "camera_number": camera_id,
                                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "accuracy_percentage": 0.0,  # Unknown confidence
                                    "face_image": encode_image_to_binary(face_image)
                                }
                                
                                # Print the data being sent to MongoDB
                                print_mongodb_data(detection_data, "camera_feed_unknown")
                                
                                # Insert into MongoDB
                                collection.insert_one(detection_data)
                                
                        except Exception as crop_error:
                            print(f"Error cropping face: {crop_error}, bbox: {bbox}, frame shape: {frame.shape}")
                            continue

            result_queue.put(result)
            data['frame_queue'].task_done()

        except Exception as e:
            print(f"Camera {camera_id} Processing error: {e}")

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
    data['video'] = cv.VideoCapture(url)
    # Read first frame to get original dimensions
    ret, first_frame = data['video'].read()
    if ret:
        data['original_height'], data['original_width'] = first_frame.shape[:2]

# Helper function to draw labels on the frame
def draw_label(img, text, pos, color, scale=0.7, thickness=2):
    """Improved text rendering for readability."""
    # Add black outline for better visibility
    cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness * 3)
    cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

# Function to detect faces in images from a folder
def detect_faces_in_folder(folder_path: str) -> List[Dict[str, Any]]:
    detected_faces = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing image: {image_path}")
            
            image = cv.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Get original dimensions
            orig_height, orig_width = image.shape[:2]
            
            # Resize image for processing
            resized_image = cv.resize(image, (desired_width, desired_height))

            # Detect faces
            faces = model.get(resized_image)
            if faces:
                print(f"Found {len(faces)} faces in {image_path}")
                embeddings = []
                face_data = []

                for face in faces:
                    bbox = face.bbox.astype(int)
                    embedding = face.embedding.astype(np.float32).reshape(1, -1)
                    embedding /= np.linalg.norm(embedding)  # Normalize

                    embeddings.append(embedding[0])
                    face_data.append((face, bbox))

                if embeddings:
                    embeddings = np.array(embeddings, dtype=np.float32)
                    distances, indices = index.search(embeddings, k=1)

                    for i, ((face, bbox), distance, idx) in enumerate(zip(face_data, distances, indices)):
                        confidence = float(distance[0])
                        matched_name = Y[idx[0]] if confidence > FACE_RECOGNITION_THRESHOLD else "Unknown"
                        
                        # Scale bounding box to the original image dimensions
                        orig_bbox = [
                            max(0, int(bbox[0] * orig_width / desired_width)),
                            max(0, int(bbox[1] * orig_height / desired_height)),
                            min(orig_width-1, int(bbox[2] * orig_width / desired_width)),
                            min(orig_height-1, int(bbox[3] * orig_height / desired_height))
                        ]
                        
                        # Check if bbox has valid dimensions
                        if orig_bbox[2] <= orig_bbox[0] or orig_bbox[3] <= orig_bbox[1]:
                            print(f"Invalid bounding box for face {i}: {orig_bbox}")
                            continue
                            
                        try:
                            # Crop the detected face from the original image
                            face_image = image[orig_bbox[1]:orig_bbox[3], orig_bbox[0]:orig_bbox[2]]
                            
                            # Check if face_image is empty
                            if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
                                print(f"Empty face image for face {i}: bbox={orig_bbox}, image shape={image.shape}")
                                continue
                                
                            # Store all faces, including unknown ones
                            detection_result = {
                                "person_name": matched_name,
                                "image_path": image_path,
                                "accuracy_percentage": confidence * 100 if matched_name != "Unknown" else 0.0,
                                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "face_image": encode_image_to_binary(face_image)
                            }
                            
                            # Print the data being sent to MongoDB
                            print_mongodb_data(detection_result, "folder_processing")
                            
                            # Add to results
                            detected_faces.append(detection_result)
                            
                            # Store in MongoDB
                            collection.insert_one(detection_result)
                            
                        except Exception as e:
                            print(f"Error processing face in {image_path}: {e}")
                            continue
            else:
                print(f"No faces detected in {image_path}")

    print(f"Total faces detected in folder: {len(detected_faces)}")
    return detected_faces

# FastAPI endpoint to get detected faces from the camera feed
@app.get("/detected-faces", response_model=List[DetectionResult])
def get_detected_faces():
    detected_faces = []
    for i in camera_data:
        data = camera_data[i]
        if data['last_result']:
            for face in data['last_result']['faces']:
                detected_faces.append({
                    "person_name": face['name'],
                    "camera_number": i,
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "accuracy_percentage": face['confidence'] * 100  # Convert to percentage
                })
    
    print(f"API request: /detected-faces - Returning {len(detected_faces)} faces")
    return detected_faces

# FastAPI endpoint to get detected faces from the photo folder
@app.get("/detected-faces-from-folder", response_model=List[DetectionResult])
def get_detected_faces_from_folder(folder_path: str):
    print(f"API request: /detected-faces-from-folder - Processing folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        raise HTTPException(status_code=404, detail="Folder not found")

    detected_faces = detect_faces_in_folder(folder_path)
    print(f"API response: Returning {len(detected_faces)} faces from folder")
    return detected_faces

# UPDATED API endpoint for the React frontend
@app.get("/api/recognition")
def get_recognition_data(cameraId: int):
    """
    API endpoint that provides recognition data in the format expected by the React frontend.
    Returns known and unknown users detected by any camera, with the most recent image for each person.
    """
    print(f"API request: /api/recognition - Camera ID: {cameraId}")
    
    # Format for frontend expected data
    known_users = []
    unknown_users = []
    
    # Track the most recent detection for each person
    known_person_latest = {}  # {name: latest_detection}
    
    # Fetch all detections from MongoDB (not just recent ones)
    all_detections = list(collection.find().sort("date_time", -1))
    
    print(f"Found {len(all_detections)} total detections in database")
    
    # Process each detection
    for detection in all_detections:
        person_name = detection.get("person_name", "Unknown")
        detection_time = detection.get("date_time", "Unknown time")
        
        # Extract face image binary data
        face_image_binary = detection.get("face_image", None)
        image_b64 = None
        
        # Convert binary image to base64 for frontend display
        if face_image_binary:
            try:
                # Base64 encode the binary image data
                image_b64 = f"data:image/jpeg;base64,{base64.b64encode(face_image_binary).decode('utf-8')}"
            except Exception as e:
                print(f"Error encoding image: {e}")
        
        # Create person data structure
        person_data = {
            "id": str(detection.get("_id")),
            "time": detection_time,
            "camera": f"Camera {detection.get('camera_number', 0)}",
            "image": image_b64 or "/unknown-user.png"  # Fallback to default if no image
        }
        
        # For known users, keep only the most recent detection per person
        if person_name != "Unknown":
            person_data["name"] = person_name
            
            # If we haven't seen this person before, or if this is a more recent detection
            if person_name not in known_person_latest:
                known_person_latest[person_name] = person_data
        else:
            # For unknown faces, add to the unknown list
            unknown_users.append(person_data)
    
    # Convert dictionary of latest known detections to list
    known_users = list(known_person_latest.values())
    
    # For unknown users, limit to most recent 10
    unknown_users = unknown_users[:10]
    
    print(f"Returning {len(known_users)} known users and {len(unknown_users)} unknown users")
    
    # Return formatted response
    return JSONResponse({
        "knownUsers": known_users,
        "unknownUsers": unknown_users
    })

# NEW function: Generator for MJPEG streaming
def generate_mjpeg(camera_id):
    """Generator function for MJPEG streaming"""
    data = camera_data[camera_id]
    if not data:
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not found\r\n'
        return
    
    while data['running']:
        # Check if video object exists and is opened
        if data['video'] is None or not data['video'].isOpened():
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not initialized or disconnected\r\n'
            time.sleep(1)
            continue
        
        try:
            # Try to read frame with error handling
            ret, frame = data['video'].read()
            if not ret:
                print(f"Failed to read frame from camera {camera_id} for streaming")
                # Yield an error frame instead of just continuing
                yield b'--frame\r\nContent-Type: text/plain\r\n\r\nFrame read error\r\n'
                time.sleep(0.1)
                continue
            
            # Resize frame to desired resolution for display
            resized_frame = cv.resize(frame, (desired_width, desired_height))
            
            # Draw latest face results if available
            if data['last_result'] and time.time() - data['last_result_time'] < 0.5:
                for face in data['last_result']['faces']:
                    bbox = face['bbox']
                    name = face['name']
                    confidence = face['confidence']
                    color = face['color']
                    
                    # Calculate bbox coordinates for the resized frame
                    resized_bbox = [
                        int(bbox[0] * desired_width / data['original_width']),
                        int(bbox[1] * desired_height / data['original_height']),
                        int(bbox[2] * desired_width / data['original_width']),
                        int(bbox[3] * desired_height / data['original_height'])
                    ]
                    
                    # Draw rectangle and label
                    cv.rectangle(resized_frame, 
                                (resized_bbox[0], resized_bbox[1]), 
                                (resized_bbox[2], resized_bbox[3]), 
                                color, 2)
                    
                    label = f"{name} ({confidence:.2f})"
                    label_pos = (resized_bbox[0], max(resized_bbox[1] - 10, 20))
                    draw_label(resized_frame, label, label_pos, color)
            
            # Display FPS overlay
            fps_text = f"Cam {camera_id} FPS: {data['display_fps']}"
            draw_label(resized_frame, fps_text, (10, 30), (0, 255, 0))
            
            # Convert the frame to JPEG bytes
            _, buffer = cv.imencode('.jpg', resized_frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
            
            # With this format, the browser will display the video stream properly
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + f"{len(frame_bytes)}".encode() + b'\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')
            
            # Store the processed frame for other potential uses
            with data['processed_frame_lock']:
                data['processed_frame'] = resized_frame.copy()
            
            # Control the frame rate for streaming (adjust as needed)
            time.sleep(0.03)  # ~30 FPS
            
        except cv.error as e:
            print(f"OpenCV error in camera {camera_id}: {e}")
            # Try to reinitialize the camera
            try:
                if data['video'] is not None:
                    data['video'].release()
                data['video'] = cv.VideoCapture(camera_id)
                yield b'--frame\r\nContent-Type: text/plain\r\n\r\nRecovering from error\r\n'
            except Exception as e2:
                print(f"Failed to reinitialize camera {camera_id}: {e2}")
                yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera error\r\n'
            time.sleep(1)
        except Exception as e:
            print(f"Unexpected error in camera {camera_id} stream: {e}")
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nStream error\r\n'
            time.sleep(1)

# NEW API endpoint: Live video stream with face detection
@app.get("/api/video-feed/{camera_id}")
async def video_feed(camera_id: int):
    """
    Streams video with face detection bounding boxes as MJPEG
    """
    if camera_id not in camera_data:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    print(f"API request: New client connected to video stream for camera {camera_id}")
    
    return StreamingResponse(
        generate_mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# NEW API endpoint: Get a single frame with face detection
@app.get("/api/video-snapshot/{camera_id}")
async def video_snapshot(camera_id: int):
    """
    Returns a single frame with face detection as JPEG
    """
    if camera_id not in camera_data:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    data = camera_data[camera_id]
    if data['video'] is None:
        raise HTTPException(status_code=503, detail="Camera not initialized")
    
    # Get the latest processed frame with face detection
    with data['processed_frame_lock']:
        if data['processed_frame'] is None:
            raise HTTPException(status_code=503, detail="No frame available yet")
        frame = data['processed_frame'].copy()
    
    # Convert to JPEG
    _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )

# Main loop for processing frames
def main_loop():
    while True:
        frames = []
        for i in camera_data:
            data = camera_data[i]
            ret, frame = data['video'].read()
            if not ret:
                print(f"Failed to read frame from camera {i}")
                data['running'] = False
                continue

            # Store original dimensions
            orig_height, orig_width = frame.shape[:2]
            
            # Resize frame to desired resolution for display
            resized_frame = cv.resize(frame, (desired_width, desired_height))

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

                result_orig_width = data['last_result'].get('orig_width', orig_width)
                result_orig_height = data['last_result'].get('orig_height', orig_height)

                for face in data['last_result']['faces']:
                    bbox = face['bbox']
                    name = face['name']
                    confidence = face['confidence']
                    color = face['color']
                    
                    # Draw rectangle and label
                    cv.rectangle(resized_frame, 
                                (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), 
                                color, 2)

                    label = f"{name} ({confidence:.2f})"
                    label_pos = (bbox[0], max(bbox[1] - 10, 20))  # Ensure label is visible
                    draw_label(resized_frame, label, label_pos, color)

            # Display FPS overlay
            fps_text = f"Cam {i} Display FPS: {data['display_fps']} | Processing FPS: {data['processing_fps']}"
            draw_label(resized_frame, fps_text, (10, 30), (0, 255, 0))

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
            cv.imshow('Combined Camera Feed', combined_frame)

        if cv.waitKey(1) & 0xFF in (ord('q'), 27):
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

    cv.destroyAllWindows()

# Run the main loop in a separate thread
threading.Thread(target=main_loop, daemon=True).start()

# Print startup message
print("="*70)
print("Face Detection System Started")
print("="*70)
print("MongoDB Connection: mongodb://localhost:27017/")
print("Database: face_detection_db")
print("Collection: detected_faces")
print(f"Deduplication window: {DEDUPLICATION_WINDOW_MINUTES} minutes")
print("-"*70)
print("API running at http://0.0.0.0:8001")
print("Available endpoints:")
print("  - GET /detected-faces")
print("  - GET /detected-faces-from-folder?folder_path=<path>")
print("  - GET /api/recognition?cameraId=<camera_id>")
print("  - GET /api/video-feed/{camera_id}")
print("  - GET /api/video-snapshot/{camera_id}")
print("-"*70)
print("Press 'q' or 'ESC' in the video window to exit")
print("="*70)

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    import io  # Required for BytesIO in the snapshot endpoint
    uvicorn.run(app, host="0.0.0.0", port=8001)

    