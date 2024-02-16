pip install opencv-python
import cv2

class DistanceEstimator:
    def __init__(self, average_car_size_meters):
        self.average_car_size_meters = average_car_size_meters

    def calculate_distance_to_car(self, detected_car_size_pixels):
        # Calculate distance using simple proportion
        distance_to_car = (self.average_car_size_meters * detected_car_size_pixels) / self.average_car_size_pixels
        return distance_to_car

    @staticmethod
    def detect_cars(image):
        # Load pre-trained car detection model
        car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect cars in the image
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)

        # Return list of bounding boxes
        return cars
