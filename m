import torch
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import pyttsx3
from threading import Thread

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
        
        # Load YOLOv5 model
        print("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Set model parameters
        self.model.conf = 0.5  # Confidence threshold
        self.model.iou = 0.45  # NMS IOU threshold
        self.model.classes = [0, 56, 57, 58, 73, 62, 24, 74]  # Filter for classroom objects
        
        # Class labels for classroom objects
        self.labels = {
            0: 'person',
            56: 'chair',
            57: 'desk',
            58: 'blackboard',
            73: 'book',
            62: 'laptop',
            24: 'backpack',
            74: 'clock'
        }
        
        self.last_description_time = 0
        self.description_cooldown = 5  # Seconds between descriptions
        
    def detect_objects(self, frame):
        """Detect objects in the frame using YOLOv5"""
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = self.model(frame_rgb)
        
        # Process detections
        detected_objects = {}
        
        # Get detections
        detections = results.xyxy[0]  # Get detection boxes in xyxy format
        
        # Count detected objects
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            class_id = int(cls)
            if class_id in self.labels:
                label = self.labels[class_id]
                detected_objects[label] = detected_objects.get(label, 0) + 1
        
        return detected_objects, detections
    
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
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on the frame"""
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            class_id = int(cls)
            
            if class_id in self.labels:
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{self.labels[class_id]}: {int(conf * 100)}%"
                cv2.putText(frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def save_frame(self, frame, detected_objects):
        """Save frame with timestamp and detection info"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        
        # Save detection info
        with open(f"detection_{timestamp}.txt", "w") as f:
            f.write(self.generate_description(detected_objects))
        
        print(f"Saved detection to {filename}")
    
    def run(self):
        """Main loop for continuous detection"""
        try:
            while True:
                # Capture frame
                frame = self.camera.capture_array()
                
                # Detect objects
                detected_objects, detections = self.detect_objects(frame)
                
                # Generate and speak description if enough time has passed
                current_time = time.time()
                if current_time - self.last_description_time > self.description_cooldown:
                    description = self.generate_description(detected_objects)
                    print("Scene Description:", description)
                    self.speak_description(description)
                    self.last_description_time = current_time
                
                # Draw detections on frame
                annotated_frame = self.draw_detections(frame, detections)
                
                # Display frame
                cv2.imshow('Classroom Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_frame(annotated_frame, detected_objects)
                
        except KeyboardInterrupt:
            print("Stopping detection...")
        
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize and run detector
    detector = ClassroomDetectorPi()
    detector.run()
