"""
Gradio Web Interface for Road Damage Detection
"""
import gradio as gr
from PIL import Image
import numpy as np

# TODO: Initialize detector
# from src.inference import RoadDamageDetector
# detector = RoadDamageDetector(
#     model_paths=[
#         'models/yolov8n_640.pt',
#         'models/yolov8s_640.pt',
#         'models/yolov8s_1024.pt'
#     ],
#     mode='accurate'
# )

DAMAGE_CLASSES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Other Corruption"
}


def predict_damage(image, mode, confidence_threshold):
    """
    Predict road damage in the image

    Args:
        image: Input image
        mode: Detection mode
        confidence_threshold: Minimum confidence

    Returns:
        Annotated image and detection results
    """
    if image is None:
        return None, "Please upload an image"

    # TODO: Implement actual detection
    # results = detector.predict(image)
    # annotated = detector.visualize(results)

    # Placeholder
    results_text = """
    ## Detection Results

    **Mode**: {mode}
    **Confidence Threshold**: {conf}

    ### Detected Damages:
    - No detections (model not loaded)

    *Upload model weights to enable detection*
    """.format(mode=mode, conf=confidence_threshold)

    return image, results_text


# Create Gradio interface
with gr.Blocks(title="Road Damage Detection") as demo:
    gr.Markdown("""
    # 🛣️ Road Damage Detection System

    **State-of-the-art automated road damage detection using Multi-Resolution YOLOv8 Ensemble**

    Achieving **66.18% mAP@50** on RDD2022 benchmark
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Road Image")

            mode = gr.Radio(
                choices=["fast", "balanced", "accurate"],
                value="accurate",
                label="Detection Mode",
                info="Fast: 20ms, Balanced: 100ms, Accurate: 150ms"
            )

            confidence = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )

            submit_btn = gr.Button("🔍 Detect Damage", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Detection Results")
            output_text = gr.Markdown()

    # Examples
    gr.Examples(
        examples=[
            ["data/sample_images/United_States_003765.jpg", "accurate", 0.25],
            ["data/sample_images/India_003920.jpg", "balanced", 0.25],
            ["data/sample_images/Japan_004582.jpg", "fast", 0.25],
        ],
        inputs=[input_image, mode, confidence],
        label="Sample Images"
    )

    submit_btn.click(
        fn=predict_damage,
        inputs=[input_image, mode, confidence],
        outputs=[output_image, output_text]
    )

    gr.Markdown("""
    ---
    ### About

    This system uses a multi-resolution ensemble of three YOLOv8 models with Weighted Boxes Fusion to achieve state-of-the-art accuracy in detecting 5 types of road damage:

    1. **Longitudinal Crack** - Linear cracks parallel to road direction
    2. **Transverse Crack** - Linear cracks perpendicular to road direction
    3. **Alligator Crack** - Interconnected cracks forming alligator skin pattern
    4. **Pothole** - Bowl-shaped holes in pavement surface
    5. **Other Corruption** - Other types of road surface deterioration

    **Paper/Patent**: Patent Pending | **GitHub**: [Repository](https://github.com/yourusername/road-damage-detection)
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
