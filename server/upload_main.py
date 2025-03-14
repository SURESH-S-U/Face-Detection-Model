from flask import Flask, request, jsonify
from flask_cors import CORS
import insightface
import cv2
import numpy as np
import faiss
import tempfile
import os
import time
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load the face recognition model
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))
print("\n✅ Model loaded successfully!")

# FAISS index for face embeddings
embedding_dim = 512
index = faiss.IndexFlatIP(embedding_dim)
face_database = {}  # Maps index IDs to identities

# Path to your saved embeddings
EMBEDDINGS_PATH = "labels.pkl"
FAISS_INDEX_PATH = "face_index.faiss"

# Load existing embeddings if available
def load_known_embeddings():
    global index, face_database
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        try:
            # Load FAISS index
            index = faiss.read_index(FAISS_INDEX_PATH)
            
            # Load face database (mapping between index positions and identities)
            with open(EMBEDDINGS_PATH, 'rb') as f:
                face_database = pickle.load(f)
                
            print(f"✅ Loaded {index.ntotal} known face embeddings")
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    return False

# Initialize embeddings
load_known_embeddings()

# Face trackers for transient face tracking
face_trackers = {}
GRACE_PERIOD = 15  # Seconds to keep tracking a face after it disappears
IOU_THRESHOLD = 0.5  # IoU threshold for matching faces between frames

def bbox_iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    box1 = [float(x) for x in box1]
    box2 = [float(x) for x in box2]

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

def normalize(emb):
    """Normalize embeddings for cosine similarity."""
    if len(emb.shape) == 1:
        return np.expand_dims(emb / np.linalg.norm(emb), axis=0)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)

def detect_and_embed(img):
    """Detects faces in an image and extracts embeddings."""
    faces = model.get(img)
    if not faces:
        return [], None  # No faces detected

    # Sort faces based on size (largest face first)
    faces = sorted(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
    embeddings = np.array([face.embedding for face in faces], dtype=np.float32)
    return faces, embeddings

def search_face(img):
    """Search for faces in an image using FAISS, with improved tracking for transient faces."""
    global face_trackers

    faces, test_embeddings = detect_and_embed(img)
    if test_embeddings is None:
        # No faces detected in this frame, but keep track of previously known faces
        current_time = time.time()
        results = []

        # Check for recently seen faces and include them with "tracking" label
        for track_id, info in list(face_trackers.items()):
            if current_time - info["last_seen"] < GRACE_PERIOD:
                results.append({
                    "bbox": info["bbox"], 
                    "status": "tracking", 
                    "confidence": info["score"],
                    "name": info.get("name", "unknown")
                })
            else:
                # Remove old tracks
                face_trackers.pop(track_id, None)

        return results

    # Faces were detected, run recognition
    norm_test_embeddings = normalize(test_embeddings)
    
    # Check if index is empty
    if index.ntotal == 0:
        # No known faces in database yet
        results = []
        current_time = time.time()
        
        for idx, face in enumerate(faces):
            bbox = face.bbox.tolist()
            new_track_id = str(len(face_trackers) + 1)
            
            # All faces are unknown since we have no reference database
            face_trackers[new_track_id] = {
                "name": "unknown",
                "last_seen": current_time,
                "bbox": bbox,
                "embedding": norm_test_embeddings[idx],
                "score": 0.0
            }
            results.append({"bbox": bbox, "status": "unknown", "confidence": 0.0, "name": "unknown"})
            
        return results
    
    # Search against known faces
    D, I = index.search(norm_test_embeddings, 1)  # Search for the closest match

    results = []
    current_time = time.time()

    # Process detected faces
    for idx, (face, embedding, best_match_idx, best_score) in enumerate(zip(faces, norm_test_embeddings, I[:, 0], D[:, 0])):
        bbox = face.bbox.tolist()
        matched_track_id = None

        # Try to match with existing tracks using IOU
        for track_id, info in face_trackers.items():
            iou = bbox_iou(bbox, info["bbox"])
            if iou > IOU_THRESHOLD:
                matched_track_id = track_id
                break

        # Get the identity if the face is recognized
        face_identity = "unknown"
        if best_match_idx != -1 and best_score > 0.5 and best_match_idx in face_database:
            face_identity = face_database[best_match_idx]

        # If this face matches a known track
        if matched_track_id is not None:
            track_info = face_trackers[matched_track_id]

            # If face is recognized confidently
            if best_match_idx != -1 and best_score > 0.5:
                track_info["name"] = face_identity
                track_info["last_seen"] = current_time
                track_info["bbox"] = bbox
                track_info["embedding"] = embedding
                track_info["score"] = best_score
                results.append({
                    "bbox": bbox, 
                    "status": "known", 
                    "confidence": float(best_score),
                    "name": face_identity
                })
            else:
                # Face not recognized confidently, but we have a tracking history
                track_info["last_seen"] = current_time
                track_info["bbox"] = bbox
                track_info["embedding"] = embedding
                results.append({
                    "bbox": bbox, 
                    "status": "tracking", 
                    "confidence": track_info["score"],
                    "name": track_info.get("name", "unknown")
                })
        else:
            # New face detected that doesn't match any existing tracks
            new_track_id = str(len(face_trackers) + 1)

            if best_match_idx != -1 and best_score > 0.4:
                # Confidently recognized new face
                face_trackers[new_track_id] = {
                    "name": face_identity,
                    "last_seen": current_time,
                    "bbox": bbox,
                    "embedding": embedding,
                    "score": best_score
                }
                results.append({
                    "bbox": bbox, 
                    "status": "known", 
                    "confidence": float(best_score),
                    "name": face_identity
                })
            else:
                # Unrecognized new face
                face_trackers[new_track_id] = {
                    "name": "unknown",
                    "last_seen": current_time,
                    "bbox": bbox,
                    "embedding": embedding,
                    "score": 0.0
                }
                results.append({
                    "bbox": bbox, 
                    "status": "unknown", 
                    "confidence": 0.0,
                    "name": "unknown"
                })

    # Clean up old tracks
    for track_id in list(face_trackers.keys()):
        if current_time - face_trackers[track_id]["last_seen"] > GRACE_PERIOD:
            face_trackers.pop(track_id, None)

    return results

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Clear previous face trackers for new video
    global face_trackers
    face_trackers = {}

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        file.save(temp_file.name)
        video_path = temp_file.name

    # Process the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        os.remove(video_path)  # Clean up if we couldn't open the video
        return jsonify({"error": "Could not open video file"}), 400

    results = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30 fps if we can't get the value
    timestamp = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame to improve performance
            if frame_count % 5 == 0:
                # Calculate timestamp in seconds
                timestamp = frame_count / fps
                
                # Process frame for face recognition
                frame_results = search_face(frame)
                
                # Add timestamp to results
                for result in frame_results:
                    result["timestamp"] = f"{int(timestamp // 60)}:{int(timestamp % 60):02d}"
                    
                results.extend(frame_results)
                
            frame_count += 1
    except Exception as e:
        print(f"Error processing video: {e}")
        cap.release()
        os.remove(video_path)
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500

    cap.release()
    os.remove(video_path)  # Clean up the temporary file

    # Count unique faces by tracking IDs to avoid duplicates
    known_faces = {}
    unknown_count = 0
    
    seen_faces = set()
    processed_results = []
    
    for r in results:
        # Create a unique identifier for this face detection
        # Use name and timestamp to better identify unique faces
        face_key = f"{r['name']}_{r['timestamp']}"
        
        # Only count each unique face once
        if face_key not in seen_faces:
            seen_faces.add(face_key)
            
            if r["status"] == "known":
                known_faces[r["name"]] = known_faces.get(r["name"], 0) + 1
            elif r["status"] == "unknown":
                unknown_count += 1
                
            processed_results.append(r)

    # Format final response
    response = {
        "known_faces": known_faces,
        "known_count": sum(known_faces.values()),
        "unknown_count": unknown_count,
        "results": processed_results  # Return processed, deduplicated results
    }
    
    return jsonify(response)

@app.route('/register', methods=['POST'])
def register_face():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    if 'name' not in request.form:
        return jsonify({"error": "No name provided"}), 400

    file = request.files['file']
    name = request.form['name']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        file.save(temp_file.name)
        image_path = temp_file.name

    # Load and process the image
    img = cv2.imread(image_path)
    if img is None:
        os.remove(image_path)
        return jsonify({"error": "Could not read image file"}), 400
        
    # Detect faces and extract embeddings
    faces, embeddings = detect_and_embed(img)
    os.remove(image_path)  # Clean up the temporary file
    
    if embeddings is None or len(faces) == 0:
        return jsonify({"error": "No face detected in the image"}), 400
        
    # Add the face embedding to the index (use the largest face if multiple detected)
    global index, face_database
    face_id = index.ntotal  # Next available index
    
    # Normalize the embedding before adding to the index
    normalized_embedding = normalize(embeddings[0])
    
    # Add the embedding to the index
    index.add(normalized_embedding)
    face_database[face_id] = name
    
    # Save the updated index and database
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(face_database, f)
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return jsonify({"error": f"Error saving face: {str(e)}"}), 500
    
    return jsonify({
        "success": True,
        "message": f"Face for {name} registered successfully",
        "total_known_faces": index.ntotal
    })

@app.route('/reset', methods=['POST'])
def reset_database():
    global index, face_database
    
    # Reset the index and database
    index = faiss.IndexFlatIP(embedding_dim)
    face_database = {}

    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(face_database, f)
    except Exception as e:
        print(f"Error resetting database: {e}")
        return jsonify({"error": f"Error resetting database: {str(e)}"}), 500
    
    return jsonify({
        "success": True,
        "message": "Face database reset successfully"
    })

if __name__ == '__main__':
    app.run(debug=True)