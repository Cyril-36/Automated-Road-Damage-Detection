"""
Flask REST API for Road Damage Detection
"""
import sys
import os
import io
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Add project root to path so we can import src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------
detector = None

MODEL_PATHS = [
    'models/yolov8n_640_best.pt',
    'models/yolov8s_640_best.pt',
    'models/yolov8s_1024_best.pt',
]


def get_detector():
    """Lazy-load the detector on first request."""
    global detector
    if detector is not None:
        return detector

    # Resolve paths relative to project root
    root = os.path.join(os.path.dirname(__file__), '..')
    resolved = [os.path.join(root, p) for p in MODEL_PATHS]
    available = [p for p in resolved if os.path.isfile(p)]

    if not available:
        return None

    from src.inference import RoadDamageDetector
    detector = RoadDamageDetector(
        model_paths=available,
        config_path=os.path.join(root, 'config', 'ensemble_config.yaml'),
        mode='accurate',
    )
    return detector


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    det = get_detector()
    return jsonify({
        'status': 'healthy',
        'service': 'road-damage-detection',
        'version': '1.0.0',
        'models_loaded': det is not None,
        'num_models': len(det.ensemble.models) if det else 0,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict road damage from an uploaded image.

    Form fields:
        image   – image file (required)
        mode    – 'fast' | 'balanced' | 'accurate' (default: accurate)
        confidence – float 0-1 (default: 0.25)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        mode = request.form.get('mode', 'accurate')
        conf = float(request.form.get('confidence', 0.25))

        image = Image.open(image_file.stream).convert('RGB')

        det = get_detector()
        if det is None:
            return jsonify({
                'error': 'No model weights found. Download them to the models/ directory first.',
            }), 503

        detections = det.predict(image, mode=mode, conf_threshold=conf)

        # Optionally return annotated image as base64
        annotated = det.visualize(image, detections)
        buf = io.BytesIO()
        annotated.save(buf, format='JPEG', quality=85)
        annotated_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'detections': detections,
            'count': len(detections),
            'image_size': list(image.size),
            'mode': mode,
            'annotated_image': annotated_b64,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint."""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400

        images = request.files.getlist('images')
        mode = request.form.get('mode', 'accurate')
        conf = float(request.form.get('confidence', 0.25))

        det = get_detector()
        if det is None:
            return jsonify({'error': 'No model weights found.'}), 503

        results = []
        for img_file in images:
            image = Image.open(img_file.stream).convert('RGB')
            detections = det.predict(image, mode=mode, conf_threshold=conf)
            results.append({
                'filename': img_file.filename,
                'detections': detections,
                'count': len(detections),
            })

        return jsonify({
            'results': results,
            'total_images': len(images),
            'mode': mode,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
