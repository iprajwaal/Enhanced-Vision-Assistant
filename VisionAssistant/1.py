import os
import math
import queue
import threading
import signal
import vertexai
from google.cloud import vision_v1
from google.cloud import texttospeech
import cv2
import time
import numpy as np
import pygame
from vertexai.preview.generative_models import GenerativeModel
from scipy.spatial import distance
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class NavigationPriority(Enum):
    CRITICAL = 1    # Immediate collision risk
    HIGH = 2        # Hazardous objects or close obstacles
    MEDIUM = 3      # Notable obstacles at moderate distance
    LOW = 4         # Background information
    IGNORE = 5      # Not worth mentioning


@dataclass
class DetectedObject:
    name: str
    bbox: List[vision_v1.NormalizedVertex]
    confidence: float
    depth_estimate: float
    priority: NavigationPriority
    trajectory: Tuple[float, float] = (0, 0)  # x, y movement
    last_seen: float = time.time()


class MovementTracker:
    def __init__(self):
        self.movement_history = []
        self.last_position = None
        
    def update_position(self, position):
        if self.last_position:
            movement = {
                'from': self.last_position,
                'to': position,
                'timestamp': time.time()
            }
            self.movement_history.append(movement)
        self.last_position = position
        
    def get_recent_movement(self):
        # Return last 5 movements
        return self.movement_history[-5:] if self.movement_history else []
    
class NavigationContext:
    def __init__(self):
        self.known_obstacles: Dict[str, DetectedObject] = {}
        self.danger_zones: List[Tuple[float, float, float]] = []  # x, y, radius
        self.safe_paths: List[Tuple[float, float]] = []
        self.user_movement = {"speed": 0, "direction": 0}
        self.last_guidance_time = 0
        self.environment_type = "unknown"
        
    def update(self, new_objects: List[DetectedObject]):
        current_time = time.time()
        # Update known obstacles with new information
        for obj in new_objects:
            if obj.name in self.known_obstacles:
                old_obj = self.known_obstacles[obj.name]
                # Calculate object movement
                old_center = self.get_object_center(old_obj.bbox)
                new_center = self.get_object_center(obj.bbox)
                trajectory = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                obj.trajectory = trajectory
            self.known_obstacles[obj.name] = obj
            
        # Clean up old obstacles
        self.known_obstacles = {k: v for k, v in self.known_obstacles.items() 
                              if current_time - v.last_seen < 10}  # 10 second timeout
        
        # Update danger zones based on current obstacles
        self.update_danger_zones()
        
    def get_object_center(self, bbox):
        return ((bbox[0].x + bbox[2].x) / 2, (bbox[0].y + bbox[2].y) / 2)
        
    def update_danger_zones(self):
        self.danger_zones = []
        for obj in self.known_obstacles.values():
            center = self.get_object_center(obj.bbox)
            # Calculate danger zone radius based on object size and type
            size = (obj.bbox[2].x - obj.bbox[0].x) * (obj.bbox[2].y - obj.bbox[0].y)
            radius = size * (1.5 if obj.priority == NavigationPriority.CRITICAL else 1.0)
            self.danger_zones.append((center[0], center[1], radius))

class AgentMind:
    def __init__(self):
        self.context = NavigationContext()
        self.hazard_weights = {
            'stairs': 1.0, 'hole': 1.0, 'edge': 1.0,
            'glass': 0.9, 'knife': 0.9, 'fire': 1.0,
            'chair': 0.7, 'table': 0.7, 'person': 0.6,
            'wall': 0.8, 'door': 0.6, 'furniture': 0.7
        }
        
    def analyze_scene(self, objects: List[DetectedObject]) -> Dict:
        """Analyze the scene and make intelligent decisions about navigation"""
        analysis = {
            'immediate_threats': [],
            'potential_hazards': [],
            'safe_paths': [],
            'guidance_priority': NavigationPriority.LOW,
            'recommended_action': None
        }
        
        # Update navigation context
        self.context.update(objects)
        
        # Identify immediate threats
        for obj in objects:
            if self.is_immediate_threat(obj):
                analysis['immediate_threats'].append(obj)
                analysis['guidance_priority'] = NavigationPriority.CRITICAL
                
        # Find safe paths
        safe_paths = self.identify_safe_paths(objects)
        if safe_paths:
            analysis['safe_paths'] = safe_paths
            
        # Determine recommended action
        analysis['recommended_action'] = self.determine_best_action(analysis)
        
        return analysis
    
    def is_immediate_threat(self, obj: DetectedObject) -> bool:
        """Determine if an object poses an immediate threat"""
        bbox = obj.bbox
        height = bbox[2].y - bbox[0].y
        width = bbox[2].x - bbox[0].x
        center_x = (bbox[0].x + bbox[2].x) / 2
        
        conditions = [
            height > 0.4 and 0.3 < center_x < 0.7,  # Large object directly ahead
            obj.name.lower() in self.hazard_weights and self.hazard_weights[obj.name.lower()] > 0.8,
            obj.depth_estimate < 1.5 and width > 0.3,  # Very close wide object
            any(self.is_moving_towards_user(o) for o in [obj])  # Object moving towards user
        ]
        
        return any(conditions)
    
    def is_moving_towards_user(self, obj: DetectedObject) -> bool:
        """Check if object is moving towards the user"""
        if not obj.trajectory:
            return False
        return obj.trajectory[1] > 0.1  # Threshold for significant movement
    
    def identify_safe_paths(self, objects: List[DetectedObject]) -> List[str]:
        """Identify safe navigation paths"""
        safe_paths = []
        sectors = {
            'left': {'clear': True, 'score': 0},
            'center': {'clear': True, 'score': 0},
            'right': {'clear': True, 'score': 0}
        }
        
        for obj in objects:
            center_x = (obj.bbox[0].x + obj.bbox[2].x) / 2
            if center_x < 0.33:
                sectors['left']['clear'] = False
                sectors['left']['score'] += self.calculate_obstacle_score(obj)
            elif center_x < 0.66:
                sectors['center']['clear'] = False
                sectors['center']['score'] += self.calculate_obstacle_score(obj)
            else:
                sectors['right']['clear'] = False
                sectors['right']['score'] += self.calculate_obstacle_score(obj)
        
        for sector, data in sectors.items():
            if data['clear'] or data['score'] < 0.5:
                safe_paths.append(sector)
        
        return safe_paths
    
    def calculate_obstacle_score(self, obj: DetectedObject) -> float:
        """Calculate how much an obstacle blocks a path"""
        size = (obj.bbox[2].y - obj.bbox[0].y) * (obj.bbox[2].x - obj.bbox[0].x)
        hazard_weight = self.hazard_weights.get(obj.name.lower(), 0.5)
        depth_factor = 1 / max(obj.depth_estimate, 0.1)
        return size * hazard_weight * depth_factor
    
    def determine_best_action(self, analysis: Dict) -> str:
        """Determine the best action based on scene analysis"""
        if analysis['immediate_threats']:
            threat = analysis['immediate_threats'][0]
            center_x = (threat.bbox[0].x + threat.bbox[2].x) / 2
            return "Stop and step right" if center_x < 0.5 else "Stop and step left"
        
        if analysis['safe_paths']:
            if 'center' in analysis['safe_paths']:
                return "Proceed straight ahead carefully"
            elif 'left' in analysis['safe_paths']:
                return "Turn slightly left and proceed"
            elif 'right' in analysis['safe_paths']:
                return "Turn slightly right and proceed"
        
        return "Stop and wait for assistance"
    
class EnhancedVisionAssistant:
    def __init__(self):
        # Initialize pygame mixer for audio
        pygame.mixer.init(buffer=512)
        
        # Configuration
        self.PROJECT_ID = 'central-rush-447806-r8'
        self.REGION = 'us-central1'
        self.CREDENTIALS_PATH = '/Users/prajwal/Developer/Enhanced-Vision-Assistant/credentials/key.json'
        
        # Initialize state variables
        self.running = False
        self.is_speaking = False
        self.cap = None
        self.audio_thread = None
        self.previous_guidance = None
        self.movement_tracker = MovementTracker()
        
        # Initialize queues and clients
        self.audio_queue = queue.PriorityQueue()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.CREDENTIALS_PATH
        vertexai.init(project=self.PROJECT_ID, location=self.REGION)
        self.vision_client = vision_v1.ImageAnnotatorClient()
        self.speech_client = texttospeech.TextToSpeechClient()
        self.model = GenerativeModel("gemini-pro")
        
        # Initialize the agent mind
        self.agent = AgentMind()
        
        # Configuration parameters
        self.DETECTION_INTERVAL = 3  # seconds
        self.MIN_OBJECT_SIZE = 0.1

    def detect_objects_with_depth(self, frame):
        """Detect objects and estimate their depth"""
        try:
            success, buffer = cv2.imencode('.jpg', frame)
            content = buffer.tobytes()
            
            image = vision_v1.Image(content=content)
            features = [
                vision_v1.Feature(type=vision_v1.Feature.Type.OBJECT_LOCALIZATION),
                vision_v1.Feature(type=vision_v1.Feature.Type.LABEL_DETECTION)
            ]
            
            request = vision_v1.AnnotateImageRequest(image=image, features=features)
            response = self.vision_client.annotate_image(request=request)
            
            detected_objects = []
            for obj in response.localized_object_annotations:
                bbox = obj.bounding_poly.normalized_vertices
                height = bbox[2].y - bbox[0].y
                width = bbox[2].x - bbox[0].x
                
                if height * width > self.MIN_OBJECT_SIZE:
                    depth_estimate = 1 / (height * width)
                    priority = self.calculate_priority(obj, bbox, depth_estimate)
                    
                    detected_obj = DetectedObject(
                        name=obj.name,
                        bbox=bbox,
                        confidence=obj.score,
                        depth_estimate=depth_estimate,
                        priority=priority
                    )
                    detected_objects.append(detected_obj)
            
            return detected_objects
            
        except Exception as e:
            print(f"Object detection error: {e}")
            return []

    def calculate_priority(self, obj, bbox, depth_estimate):
        """Calculate priority based on object properties"""
        if self.agent.is_immediate_threat(DetectedObject(
            name=obj.name, bbox=bbox, confidence=obj.score,
            depth_estimate=depth_estimate, priority=NavigationPriority.LOW
        )):
            return NavigationPriority.CRITICAL
            
        height = bbox[2].y - bbox[0].y
        center_x = (bbox[0].x + bbox[2].x) / 2
        
        if height > 0.4 or obj.name.lower() in self.agent.hazard_weights:
            return NavigationPriority.HIGH
        elif 0.3 < center_x < 0.7 and height > 0.2:
            return NavigationPriority.MEDIUM
        elif height > 0.1:
            return NavigationPriority.LOW
        return NavigationPriority.IGNORE

    def generate_smart_guidance(self, objects):
        """Generate intelligent guidance using scene analysis"""
        try:
            if not objects:
                return None

            # Generate scene description
            scene_description = []
            for obj in objects:
                description = self.generate_enhanced_object_description(obj, obj.bbox)
                scene_description.append(description)

            # Update context
            context = {
                'previous_guidance': self.previous_guidance,
                'movement_history': self.movement_tracker.get_recent_movement(),
                'environment_type': 'indoor',  # This could be detected
                'current_speed': 'normal',
                'recent_obstacles': [obj.name for obj in objects],
                'safe_zones': [],
                'last_guidance_time': time.time()
            }

            # Generate and send prompt to AI model
            prompt = self.generate_agent_prompt(scene_description, context)
            response = self.model.generate_content(prompt)

            # Store guidance for future context
            self.previous_guidance = response.text
            return response.text

        except Exception as e:
            print(f"Guidance generation error: {e}")
            return self.generate_fallback_guidance(objects)

    def generate_enhanced_object_description(self, obj, bbox):
        """Generate detailed object description"""
        center_x = (bbox[0].x + bbox[2].x) / 2
        position = self.calculate_relative_position(center_x)
        distance = self.estimate_distance(bbox[2].y - bbox[0].y, bbox[2].x - bbox[0].x)
        
        description = f"{obj.name} {position} at {distance}"
        return description

    def calculate_relative_position(self, center_x):
        """Calculate relative position of object"""
        if center_x < 0.2:
            return "far left"
        elif center_x < 0.4:
            return "to your left"
        elif center_x < 0.6:
            return "directly ahead"
        elif center_x < 0.8:
            return "to your right"
        else:
            return "far right"

    def estimate_distance(self, height, width):
        """Estimate distance using object dimensions"""
        area = height * width
        if area > 0.5:
            return "very close"
        elif area > 0.3:
            return "close"
        elif area > 0.1:
            return "moderate distance"
        else:
            return "far ahead"

    def generate_agent_prompt(self, scene_description, context):
        """Generate a sophisticated prompt for the AI agent"""
        prompt = f"""You are an intelligent navigation assistant for a visually impaired person. Your role is to be their eyes and ensure their safety while helping them navigate their environment. Think carefully through each step before providing guidance.

            Current Scene Analysis:
            1. Detected Objects: {', '.join(scene_description)}
            2. Previous Context: {context.get('previous_guidance', 'No previous guidance')}
            3. Movement History: {context.get('movement_history', 'Starting navigation')}
            4. Environment Type: {context.get('environment_type', 'Unknown')}

            Step-by-step Thinking Process:
            1. First, analyze immediate safety threats:
            - Are there any obstacles in imminent collision path?
            - Are there any hazardous objects nearby?
            - Is there any moving object approaching?

            2. Then, evaluate navigation options:
            - Which paths are completely clear?
            - What is the safest direction to move?
            - Are there any stable objects that could serve as landmarks?

            3. Consider environmental context:
            - Is this an indoor or outdoor space?
            - Are there any recognizable features for orientation?
            - Are there any potential changes in elevation (steps, curbs)?

            4. Think about user comfort:
            - How can they move most confidently?
            - What landmarks can help them maintain orientation?
            - How urgent is the guidance needed?

            Based on this analysis, provide:
            1. A primary safety alert if needed (most urgent threats)
            2. Clear directional guidance (where to move)
            3. Contextual information (what to expect in that direction)

            Requirements for your response:
            - Use natural, conversational language suitable for text-to-speech
            - Be concise but thorough (maximum 3 sentences)
            - Start with any urgent warnings
            - Use clear spatial references (left, right, ahead, behind)
            - Mention distances in practical terms (very close, nearby, ahead)
            - If relevant, include time-sensitive information (approaching object, changing conditions)

            Additional Context:
            - Person's current speed: {context.get('current_speed', 'unknown')}
            - Recent obstacles: {context.get('recent_obstacles', [])}
            - Known safe zones: {context.get('safe_zones', [])}
            - Last guidance timestamp: {context.get('last_guidance_time', 'initial')}

            Give your response in a clear, calming voice that prioritizes safety while maintaining user confidence.
            """
        return prompt
    
    def generate_fallback_guidance(self, objects):
        """Generate basic guidance when AI model fails"""
        if not objects:
            return "Path appears clear, proceed with caution."
            
        warnings = []
        for obj in objects:
            description = self.generate_enhanced_object_description(obj, obj.bbox)
            warnings.append(description)
        
        return ". ".join(warnings) + ". Proceed with caution."

    def process_audio_queue(self):
        """Process audio queue with priority handling"""
        while self.running:
            try:
                if not self.is_speaking and not self.audio_queue.empty():
                    priority, text = self.audio_queue.get(timeout=1)
                    if text:  # Only speak if there's actual guidance
                        self.speak(text, priority)
                time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")

    def speak(self, text, priority=5):
        """Enhanced text-to-speech with priority handling"""
        try:
            self.is_speaking = True
            
            # Configure voice based on priority
            speaking_rate = 1.4 if priority <= 2 else 1.2
            
            input_text = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Standard-F",
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
                pitch=1 if priority > 2 else 1.2
            )

            response = self.speech_client.synthesize_speech(
                input=input_text, voice=voice, audio_config=audio_config
            )

            # Use a temporary file with priority in name
            temp_file = f"temp_audio_p{priority}_{time.time()}.mp3"
            with open(temp_file, "wb") as out:
                out.write(response.audio_content)
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy() and self.running:
                time.sleep(0.1)
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            print(f"Speech synthesis error: {e}")
        finally:
            self.is_speaking = False

    def display_debug_frame(self, frame, objects):
        """Display frame with object annotations for debugging"""
        if objects:
            for obj in objects:
                bbox = obj.bbox
                h, w = frame.shape[:2]
                pts = np.array([[int(v.x * w), int(v.y * h)] for v in bbox], np.int32)
                
                # Color based on priority
                color = (0, 255, 0)  # Default color
                if obj.priority == NavigationPriority.CRITICAL:
                    color = (0, 0, 255)  # Red for critical
                elif obj.priority == NavigationPriority.HIGH:
                    color = (0, 165, 255)  # Orange for high priority
                
                cv2.polylines(frame, [pts], True, color, 2)
                cv2.putText(frame, f"{obj.name} (P:{obj.priority.value})", 
                          (int(bbox[0].x * w), int(bbox[0].y * h) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Enhanced Vision Assistant View', frame)

    def start(self):
        """Start the enhanced vision assistant"""
        try:
            print("Starting Enhanced Vision Assistant...")
            self.running = True
            
            # Start audio processing thread
            self.audio_thread = threading.Thread(target=self.process_audio_queue)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            last_detection_time = 0
            
            print("Press 'q' to quit")
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Perform object detection at intervals
                if current_time - last_detection_time >= self.DETECTION_INTERVAL:
                    objects = self.detect_objects_with_depth(frame)
                    
                    if objects:
                        guidance = self.generate_smart_guidance(objects)
                        if guidance:  # Only queue if there's meaningful guidance
                            # Determine priority based on scene analysis
                            analysis = self.agent.analyze_scene(objects)
                            priority = analysis['guidance_priority'].value
                            self.audio_queue.put((priority, guidance))
                    
                    last_detection_time = current_time
                
                # Display frame with annotations (debug view)
                self.display_debug_frame(frame, objects)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the assistant and cleanup"""
        print("Stopping Enhanced Vision Assistant...")
        self.running = False
        
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Cleanup resources
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.audio_thread is not None and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
        
        # Clean up temporary files
        for file in os.listdir():
            if file.startswith("temp_audio_") and file.endswith(".mp3"):
                try:
                    os.remove(file)
                except:
                    pass
        
        print("Enhanced Vision Assistant stopped.")

def main():
    assistant = EnhancedVisionAssistant()
    try:
        assistant.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        assistant.stop()


if __name__ == "__main__":
    main()