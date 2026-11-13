import numpy as np
import tensorflow as tf
import cv2

CLASSES = ['apple', 'car', 'cat', 'clock', 'fish', 'house', 'star', 'tree']

class DrawingPredictor:
    def __init__(self, model_path='models/cnn_drawing_model.keras'):
        self.model = tf.keras.models.load_model(model_path)
        self.classes = CLASSES

    def preprocess_image(self, image):
        """
        Preprocess input image to match CNN input shape (1, 28, 28, 1)
        """
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)

        if image.shape != (28, 28):
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

        # Ensure black background and white strokes (matching QuickDraw dataset)
        if np.mean(image) > 127:
            image = 255 - image

        image = image.astype('float32') / 255.0
        image = image.reshape(1, 28, 28, 1)
        return image

    def predict_top3(self, image):
        """
        Returns top 3 predictions as a list of tuples: [(class_name, confidence_percentage), ...]
        """
        processed_img = self.preprocess_image(image)
        probs = self.model.predict(processed_img, verbose=0)[0]
        
        top3_indices = np.argsort(probs)[::-1][:3]
        results = [
            (self.classes[i], round(float(probs[i]) * 100, 2))
            for i in top3_indices
        ]
        return results