import tensorflow as tf
import numpy as np
import cv2
from picamera2 import Picamera2
import time
from tflite_runtime.interpreter import Interpreter
import pyttsx3
from threading import Thread
import os

class ClassroomDetectorPi:
    def __init__(self):
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize camera
        self.camera = Picamera2()
        self.camera.preview_configuration.main.size = (640, 480)
        self.camera.preview_configuration.main.format = "RGB888"
        self.camera.configure("preview")
        self.camera.start()
        time.sleep(2)  # Allow camera to warm up
        
        # Load TFLite model
        self.model_path = 'ssdlite_mobilenet_v2.tflite'
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Define classroom objects
        self.labels = {
            1: 'person',
            62: 'chair',
            63: 'desk',
            64: 'blackboard',
            84: 'book',
            73: 'laptop',
            27: 'backpack',
            85: 'clock'
        }
        
        self.min_confidence = 0.5
        self.last_description_time = 0
        self.description_cooldown = 5  # Seconds between descriptions
        
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        input_shape = self.input_details[0]['shape'][1:3]
        resized_image = cv2.resize(image, input_shape)
        input_data = np.expand_dims(resized_image, axis=0)
        input_data = (np.float32(input_data) - 127.5) / 127.5
        return input_data
        
    def detect_objects(self, image):
        """Detect objects in the image"""
        # Preprocess image
        input_data = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Filter detections
        detected_objects = {}
        for i in range(len(scores)):
            if scores[i] > self.min_confidence:
                class_id = int(classes[i])
                if class_id in self.labels:
                    obj_name = self.labels[class_id]
                    if obj_name in detected_objects:
                        detected_objects[obj_name] += 1
                    else:
                        detected_objects[obj_name] = 1
                        
        return detected_objects, boxes, classes, scores
        
    def generate_description(self, detected_objects):
        """Generate natural language description of the scene"""
        if not detected_objects:
            return "No classroom objects detected in the scene."
        
        description_parts = []
        
        # Describe people
        if 'person' in detected_objects:
            count = detected_objects['person']
            description_parts.append(
                f"There {'is' if count == 1 else 'are'} {count} "
                f"{'person' if count == 1 else 'people'} in view"
            )
        
        # Describe furniture
        furniture = []
        if 'desk' in detected_objects:
            furniture.append(f"{detected_objects['desk']} {'desk' if detected_objects['desk'] == 1 else 'desks'}")
        if 'chair' in detected_objects:
            furniture.append(f"{detected_objects['chair']} {'chair' if detected_objects['chair'] == 1 else 'chairs'}")
        if furniture:
            description_parts.append("I can see " + " and ".join(furniture))
        
        # Describe educational equipment
        if 'blackboard' in detected_objects:
            description_parts.append("There is a blackboard")
        
        # Describe other objects
        other_objects = []
        for obj in ['laptop', 'book', 'backpack', 'clock']:
            if obj in detected_objects:
                other_objects.append(f"{detected_objects[obj]} {obj}{'s' if detected_objects[obj] > 1 else ''}")
        if other_objects:
            description_parts.append("I detect " + " and ".join(other_objects))
        
        return ". ".join(description_parts) + "."
    
    def speak_description(self, description):
        """Speak the description in a separate thread"""
        def speak():
            self.engine.say(description)
            self.engine.runAndWait()
        
        Thread(target=speak).start()
    
    def draw_detections(self, image, boxes, classes, scores):
        """Draw detection boxes on the image"""
        height, width, _ = image.shape
        for i in range(len(scores)):
            if scores[i] > self.min_confidence:
                class_id = int(classes[i])
                if class_id in self.labels:
                    # Convert normalized coordinates to pixel coordinates
                    ymin = int(boxes[i][0] * height)
                    xmin = int(boxes[i][1] * width)
                    ymax = int(boxes[i][2] * height)
                    xmax = int(boxes[i][3] * width)
                    
                    # Draw box
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{self.labels[class_id]}: {int(scores[i] * 100)}%"
                    cv2.putText(image, label, (xmin, ymin - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image
    
    def run(self):
        """Main loop for continuous detection"""
        try:
            while True:
                # Capture frame
                frame = self.camera.capture_array()
                
                # Detect objects
                detected_objects, boxes, classes, scores = self.detect_objects(frame)
                
                # Generate and speak description if enough time has passed
                current_time = time.time()
                if current_time - self.last_description_time > self.description_cooldown:
                    description = self.generate_description(detected_objects)
                    print("Scene Description:", description)
                    self.speak_description(description)
                    self.last_description_time = current_time
                
                # Draw detections on frame
                annotated_frame = self.draw_detections(frame, boxes, classes, scores)
                
                # Display frame
                cv2.imshow('Classroom Detection', annotated_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("Stopping detection...")
        
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize and run detector
    detector = ClassroomDetectorPi()
    detector.run()
