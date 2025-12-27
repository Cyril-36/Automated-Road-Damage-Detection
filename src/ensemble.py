"""
Ensemble detection logic
"""


class EnsembleDetector:
    """Ensemble detector combining multiple models"""

    def __init__(self, models, input_sizes, fusion, conf_threshold=0.25):
        """
        Initialize ensemble detector

        Args:
            models: List of model names
            input_sizes: List of input sizes for each model
            fusion: Fusion strategy object
            conf_threshold: Confidence threshold for filtering
        """
        self.models = models
        self.input_sizes = input_sizes
        self.fusion = fusion
        self.conf_threshold = conf_threshold

    def predict(self, image):
        """Run ensemble prediction"""
        # TODO: Implement ensemble prediction
        pass
