from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import insightface
import faiss
import pickle
import time
import os
import logging
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize InsightFace model
try:
    logger.info("Initializing InsightFace model...")
    model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    model.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("InsightFace model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize InsightFace model: {e}")
    raise

# Load embeddings and labels
try:
    logger.info("Loading labels and face index...")
    with open('labels.pkl', 'rb') as f:
        labels_data = pickle.load(f)
    labels = labels_data['labels']
    index = faiss.read_index('face_index.faiss')
    logger.info(f"Loaded {len(labels)} labels and face index successfully.")
except Exception as e:
    logger.error(f"Failed to load labels or face index: {e}")
    raise

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file temporarily
    os.makedirs('uploads', exist_ok=True)
    timestamp = int(time.time())
    temp_path = f"uploads/temp_video_{timestamp}.mp4"
    file.save(temp_path)

    if not os.path.exists(temp_path):
        return jsonify({"error": "Could not save uploaded file"}), 500

    # Process the video
    try:
        results = process_video(temp_path)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    known_faces = set()  # Use a set to store unique known faces
    unknown_faces = []
    all_faces = []
    face_counter = {"known_count": 0, "unknown_count": 0}
    known_faces_occurrence = {}

    frame_count = 0
    processed_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, int(fps / 4))  # Process 4 frames per second

    logger.info(f"Processing video: {total_frames} frames, {fps} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        processed_frames += 1
        if processed_frames % 10 == 0:
            logger.info(f"Processed {processed_frames} frames out of {total_frames}")

        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if max(height, width) > 640:
            scale = 640 / max(height, width)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        # Detect faces
        faces = model.get(frame)
        if not faces:
            continue

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding.astype(np.float32).reshape(1, -1)
            embedding /= np.linalg.norm(embedding)

            distances, indices = index.search(embedding, k=1)
            matched_name = labels[indices[0][0]] if distances[0][0] > 0.5 else "Unknown"  # Adjusted confidence threshold
            is_known = matched_name != "Unknown"

            # Crop face image
            face_image = frame[max(0, bbox[1]):min(frame.shape[0], bbox[3]),
                              max(0, bbox[0]):min(frame.shape[1], bbox[2])]
            _, buffer = cv2.imencode('.jpg', face_image)
            face_image_base64 = base64.b64encode(buffer).decode('utf-8')

            face_id = matched_name if is_known else f"Unknown_{hash(str(embedding.tobytes())) % 10000000}"
            face_record = {
                'face_id': face_id,
                'name': matched_name,
                'confidence': float(distances[0][0]),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'frame_position': frame_count,
                'face_image': face_image_base64
            }

            if is_known:
                if face_id not in known_faces:  # Deduplicate known faces
                    known_faces.add(face_id)
                    face_counter["known_count"] += 1
                    known_faces_occurrence[matched_name] = known_faces_occurrence.get(matched_name, 0) + 1
                    all_faces.append(face_record)
            else:
                unknown_faces.append(face_record)
                face_counter["unknown_count"] += 1
                all_faces.append(face_record)

    cap.release()
    logger.info(f"Processed {processed_frames} frames, found {len(all_faces)} faces")

    return {
        "known_count": face_counter["known_count"],
        "unknown_count": face_counter["unknown_count"],
        "known_faces": known_faces_occurrence,
        "total_processed_frames": processed_frames,
        "total_frames": total_frames,
        "results": all_faces[:100]  # Limit to first 100 faces for UI
    }

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=8000, debug=True)