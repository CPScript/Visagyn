import cv2
import numpy as np
import threading
import queue
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import time
from pathlib import Path
import urllib.request
import os
import customtkinter as ctk
from tkinter import messagebox, filedialog
import tkinter as tk
from collections import deque
import gc

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MediaPipeFaceDetector:
    def __init__(self):
        self.enabled = True
        self.face_mesh = None
        self.face_detection = None
        self.mp_available = self.setup_mediapipe()
        
    def setup_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            
            print("MediaPipe face detection initialized successfully")
            return True
        except ImportError:
            print("MediaPipe not available - install with: pip install mediapipe")
            return False
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")
            return False
    
    def detect_face_mesh(self, frame):
        if not self.enabled or not self.mp_available or not self.face_mesh:
            return None, frame
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0], frame
            return None, frame
        except Exception as e:
            print(f"MediaPipe mesh error: {e}")
            return None, frame
    
    def detect_face_bbox(self, frame):
        if not self.enabled or not self.mp_available or not self.face_detection:
            return None
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.face_detection.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                return (x, y, width, height)
            return None
        except Exception as e:
            print(f"MediaPipe bbox error: {e}")
            return None

class FastFaceTracker:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mediapipe_detector = MediaPipeFaceDetector()
        self.enabled = True
        self.use_mediapipe = self.mediapipe_detector.mp_available
        self.last_detection_time = 0
        self.detection_interval = 0.05  
        self.last_bbox = None
        self.bbox_cache_duration = 0.2  
        
        self.face_swap_enabled = False
        self.face_analyzer = None
        self.face_swapper = None
        self.target_face_embedding = None
        
        self.face_swapping_available = self.setup_face_swapping()
        
        if self.use_mediapipe:
            print("Using MediaPipe for face detection and mesh overlay")
        else:
            print("Using OpenCV for face detection (no mesh overlay available)")
        
        if self.face_swapping_available:
            print("Face swapping ready - InsightFace models loaded")
        else:
            if self.face_analyzer:
                print("Face swapping partially available - analyzer loaded but swapper failed")
                print("   Face detection works, but face swapping needs manual model download")
            else:
                print("Face swapping unavailable - install InsightFace for this feature")
                print("   Install with: pip install insightface onnxruntime")
    
    def setup_face_swapping(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import get_model
            
            print("Setting up face swapping models...")
            
            self.face_analyzer = FaceAnalysis(name='buffalo_l')
            self.face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(512, 512))
            print("Face analyzer initialized")
            
            try:
                print("Downloading face swapper model...")
                self.face_swapper = get_model('inswapper_128.onnx', download=True, download_zip=True)
                print("Face swapper initialized")
                return True
                
            except Exception as download_error:
                print(f"Face swapper download failed: {download_error}")
                print("Trying alternative download method...")
                
                try:
                    self.face_swapper = get_model('inswapper_128.onnx', download=True)
                    print("Face swapper initialized (alternative method)")
                    return True
                except Exception as alt_error:
                    print(f"Alternative download also failed: {alt_error}")
                    print("Manual solution:")
                    print("   1. Download inswapper_128.onnx manually from:")
                    print("      https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx")
                    user_path = os.path.expanduser("~")
                    model_dir = os.path.join(user_path, ".insightface", "models")
                    print(f"   2. Place it in: {model_dir}")
                    print("   3. Restart the application")
                    
                    try:
                        os.makedirs(model_dir, exist_ok=True)
                        print(f"   Model directory created: {model_dir}")
                    except Exception as dir_error:
                        print(f"   Could not create directory: {dir_error}")
                    
                    self.face_swapper = None
                    return False
            
        except ImportError as e:
            print(f"InsightFace not available for face swapping: {e}")
            print("Install with: pip install insightface onnxruntime")
            self.face_analyzer = None
            self.face_swapper = None
            return False
        except Exception as e:
            print(f"Face swapper setup error: {e}")
            self.face_analyzer = None
            self.face_swapper = None
            return False
    
    def set_target_face(self, image_path):
        try:
            if not self.face_analyzer:
                print("Face analyzer not available - InsightFace not installed")
                print("Install with: pip install insightface onnxruntime")
                return False
            
            if not os.path.exists(image_path):
                print(f"File does not exist: {image_path}")
                return False
            
            target_img = cv2.imread(image_path)
            if target_img is None:
                print(f"Could not load image: {image_path}")
                return False
            
            h, w = target_img.shape[:2]
            if w > 1024 or h > 1024:
                scale = 1024 / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                target_img = cv2.resize(target_img, (new_w, new_h))
                print(f"Resized target image from {w}x{h} to {new_w}x{new_h}")
            
            faces = None
            try:
                print("🔍 Analyzing image for faces...")
                faces = self.face_analyzer.get(target_img)
            except Exception as e:
                print(f"❌ Face analysis error: {e}")
                return False
            
            if not faces or len(faces) == 0:
                print("No faces detected in target image")
                print("Tips: Ensure face is clearly visible, well-lit, and facing forward")
                return False
            
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
                print(f"👥 Multiple faces detected, using largest one")
            
            self.target_face_embedding = faces[0]
            print(f"Target face loaded: {os.path.basename(image_path)}")
            print(f"Face confidence: {faces[0].det_score:.2f}")
            return True
                
        except Exception as e:
            print(f"Target face loading error: {e}")
            return False
    
    def swap_face_with_mesh(self, frame, face_landmarks):
        try:
            if not self.face_swap_enabled:
                return frame
            
            if not self.face_swapper or not self.target_face_embedding or not self.face_analyzer:
                if face_landmarks:
                    h, w = frame.shape[:2]
                    if not self.face_analyzer:
                        cv2.putText(frame, "FACE SWAP: INSTALL INSIGHTFACE", (10, h-40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                    elif not self.target_face_embedding:
                        cv2.putText(frame, "FACE SWAP: SELECT TARGET FACE", (10, h-40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                return frame
                
            faces = self.face_analyzer.get(frame)
            if len(faces) == 0:
                h, w = frame.shape[:2]
                cv2.putText(frame, "FACE SWAP: NO SOURCE FACE DETECTED", (10, h-40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                return frame
            
            source_face = faces[0]
            
            swapped_frame = self.face_swapper.get(
                frame, 
                source_face, 
                self.target_face_embedding, 
                paste_back=True
            )
            
            if face_landmarks:
                h, w = frame.shape[:2]
                
                mask = np.zeros((h, w), dtype=np.uint8)
                
                points = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        points.append([x, y])
                
                if len(points) > 50:
                    points = np.array(points)
                    hull = cv2.convexHull(points)
                    cv2.fillPoly(mask, [hull], 255)
                    
                    kernel = np.ones((10, 10), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    mask = cv2.GaussianBlur(mask, (31, 31), 0)
                    mask = mask.astype(np.float32) / 255.0
                    
                    mask = np.maximum(mask, 0.1)
                    mask = np.stack([mask] * 3, axis=-1)
                    
                    result = frame * (1 - mask) + swapped_frame * mask
                    return result.astype(np.uint8)
            
            return swapped_frame
            
        except Exception as e:
            print(f"Face swap error: {e}")
            h, w = frame.shape[:2]
            cv2.putText(frame, f"FACE SWAP ERROR: {str(e)[:30]}", (10, h-40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            return frame
        
    def detect_face_fast(self, frame):
        if not self.enabled:
            return None
            
        current_time = time.time()
        
        if self.use_mediapipe:
            if (self.last_bbox and 
                current_time - self.last_detection_time < self.bbox_cache_duration):
                return self.last_bbox
                
            if current_time - self.last_detection_time > self.detection_interval:
                bbox = self.mediapipe_detector.detect_face_bbox(frame)
                if bbox:
                    self.last_bbox = bbox
                    self.last_detection_time = current_time
                return self.last_bbox
            else:
                return self.last_bbox
        else:
            if current_time - self.last_detection_time > self.detection_interval:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    self.last_bbox = tuple(faces[0])
                    self.last_detection_time = current_time
                    return self.last_bbox
            return self.last_bbox
    
    def get_face_mesh(self, frame):
        if self.use_mediapipe and self.enabled:
            return self.mediapipe_detector.detect_face_mesh(frame)
        return None, frame

class ProfessionalAIProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enabled = True
        self.upscaler = None
        
        self.denoise_enabled = True
        self.sharpen_enabled = True
        self.stabilization_enabled = True
        self.fps_boost_enabled = False
        
        self.denoise_strength = 0.3
        self.sharpen_strength = 0.15
        self.upscale_factor = 1.25  
        
        self.processing_skip_counter = 0
        self.target_fps = 60
        self.quality_setting = "Balanced"
        
        self.frame_buffer = deque(maxlen=3)
        self.face_mask_buffer = deque(maxlen=3)
        self.previous_frame = None
        
        self.load_models()
    
    def load_models(self):
        model_loaded = False
        
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            model_path = "RealESRGAN_x4plus.pth"
            if not os.path.exists(model_path):
                print("Downloading RealESRGAN model...")
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                try:
                    urllib.request.urlretrieve(url, model_path)
                except:
                    print("Failed to download model")
                    self.upscaler = None
                    return False
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upscaler = RealESRGANer(
                scale=1.25,  
                model_path=model_path,
                model=model,
                tile=192, 
                tile_pad=8,
                pre_pad=0,
                half=True if self.device.type == 'cuda' else False,
                gpu_id=0 if self.device.type == 'cuda' else None
            )
            print("✓ Professional AI upscaler loaded")
            model_loaded = True
            
        except ImportError:
            print("✗ RealESRGAN not available")
            self.upscaler = None
        except Exception as e:
            print(f"✗ RealESRGAN loading error: {e}")
            self.upscaler = None
        
        return model_loaded
    
    def load_face_swapper(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import get_model
            
            self.face_analyzer = FaceAnalysis(name='buffalo_l')
            self.face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(512, 512))
            
            self.face_swapper = get_model('inswapper_128.onnx', download=True, download_zip=True)
            print("✓ Face swapping model loaded")
            return True
            
        except ImportError:
            print("✗ InsightFace not available")
            self.face_swapper = None
            self.face_analyzer = None
            return False
        except Exception as e:
            print(f"✗ Face swapper error: {e}")
            self.face_swapper = None
            self.face_analyzer = None
            return False
    
    def set_target_face(self, image_path):
        try:
            if not os.path.exists(image_path):
                print(f"File does not exist: {image_path}")
                return False
            
            target_img = cv2.imread(image_path)
            if target_img is None:
                print(f"Could not load image: {image_path}")
                return False
            
            h, w = target_img.shape[:2]
            if w > 1024 or h > 1024:
                scale = 1024 / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                target_img = cv2.resize(target_img, (new_w, new_h))
                print(f"Resized image from {w}x{h} to {new_w}x{new_h}")
            
            if not self.face_analyzer:
                print("Face analyzer not available - InsightFace not installed")
                return False
            
            faces = None
            try:
                faces = self.face_analyzer.get(target_img)
            except Exception as e:
                print(f"Face analysis error: {e}")
                try:
                    rgb_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    faces = self.face_analyzer.get(bgr_img)
                except Exception as e2:
                    print(f"Second face analysis attempt failed: {e2}")
                    return False
            
            if not faces or len(faces) == 0:
                print("No faces detected in the target image")
                print("Tips: Make sure the face is clearly visible, well-lit, and facing forward")
                return False
            
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
                print(f"Multiple faces detected, using the largest one")
            
            self.target_face_embedding = faces[0]
            print(f"✓ Target face loaded successfully: {os.path.basename(image_path)}")
            print(f"Face confidence: {faces[0].det_score:.2f}")
            return True
                
        except Exception as e:
            print(f"Target face loading error: {e}")
            return False
    
    def create_stable_face_mask(self, frame, landmarks):
        try:
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            
            if landmarks:
                points = []
                for landmark in landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        points.append([x, y])
                
                if len(points) > 50:
                    points = np.array(points)
                    hull = cv2.convexHull(points)
                    cv2.fillPoly(mask, [hull], 1.0)
                    
                    kernel = np.ones((15, 15), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    mask = cv2.GaussianBlur(mask, (21, 21), 0)
                    
                    self.face_mask_buffer.append(mask)
                    
                    if len(self.face_mask_buffer) > 1:
                        avg_mask = np.mean(self.face_mask_buffer, axis=0)
                        return np.clip(avg_mask, 0.0, 1.0)
            
            return mask
            
        except Exception as e:
            print(f"Face mask error: {e}")
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    
    def professional_denoise(self, frame, face_mask=None):
        try:
            if not self.denoise_enabled:
                return frame
            
            original = frame.copy()
            
            if face_mask is not None and np.max(face_mask) > 0:
                h, w = frame.shape[:2]
                if face_mask.shape[:2] != (h, w):
                    face_mask = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                
                face_mask_3d = np.stack([face_mask] * 3, axis=-1)
                bg_mask_3d = 1.0 - face_mask_3d
                
                bg_denoised = cv2.bilateralFilter(frame, 15, 80, 80)
                bg_denoised = cv2.medianBlur(bg_denoised, 5)
                bg_denoised = cv2.GaussianBlur(bg_denoised, (3, 3), 0)
                
                face_preserved = cv2.bilateralFilter(frame, 9, 50, 50)
                
                result = (face_preserved * face_mask_3d + bg_denoised * bg_mask_3d).astype(np.uint8)
                
                strength = self.denoise_strength
                result = cv2.addWeighted(original, 1-strength, result, strength, 0)
                
            else:
                result = cv2.bilateralFilter(frame, 12, 60, 60)
                result = cv2.medianBlur(result, 3)
                strength = self.denoise_strength * 0.7 
                result = cv2.addWeighted(original, 1-strength, result, strength, 0)
            
            return result
            
        except Exception as e:
            print(f"Denoising error: {e}")
            return frame
    
    def intelligent_sharpen(self, frame, face_mask=None):
        try:
            if not self.sharpen_enabled:
                return frame
            
            original = frame.copy()
            
            kernel = np.array([[-0.25, -0.25, -0.25],
                              [-0.25,  3.0, -0.25],
                              [-0.25, -0.25, -0.25]])
            
            sharpened = cv2.filter2D(frame, -1, kernel)
            
            gaussian = cv2.GaussianBlur(frame, (5, 5), 1.0)
            unsharp = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
            
            enhanced = cv2.addWeighted(sharpened, 0.6, unsharp, 0.4, 0)
            
            if face_mask is not None and np.max(face_mask) > 0:
                h, w = frame.shape[:2]
                if face_mask.shape[:2] != (h, w):
                    face_mask = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                
                face_mask_3d = np.stack([face_mask] * 3, axis=-1)
                bg_mask_3d = 1.0 - face_mask_3d
                
                face_sharp = cv2.addWeighted(original, 0.7, enhanced, 0.3, 0) 
                bg_sharp = cv2.addWeighted(original, 0.5, enhanced, 0.5, 0)  
                
                result = (face_sharp * face_mask_3d + bg_sharp * bg_mask_3d).astype(np.uint8)
            else:
                strength = self.sharpen_strength * 2.0 
                result = cv2.addWeighted(original, 1-strength, enhanced, strength, 0)
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Sharpening error: {e}")
            return frame
    
    def temporal_stabilization(self, frame):
        try:
            if not self.stabilization_enabled:
                return frame
            
            self.frame_buffer.append(frame.copy())
            
            if len(self.frame_buffer) >= 2:
                prev_frame = self.frame_buffer[-2]
                
                if frame.shape == prev_frame.shape:
                    alpha = 0.95 
                    stabilized = cv2.addWeighted(frame, alpha, prev_frame, 1-alpha, 0)
                    
                    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray_curr, gray_prev)
                    motion_amount = np.mean(diff)
                    
                    if motion_amount > 30: 
                        alpha = 0.98
                        stabilized = cv2.addWeighted(frame, alpha, prev_frame, 1-alpha, 0)
                    
                    return stabilized
            
            return frame
            
        except Exception as e:
            print(f"Stabilization error: {e}")
            return frame
    
    def professional_upscale(self, frame):
        try:
            if not self.enabled:
                return frame
            
            max_pixels = 640 * 480
            if frame.shape[0] * frame.shape[1] > max_pixels:
                height, width = frame.shape[:2]
                new_width = int(width * self.upscale_factor)
                new_height = int(height * self.upscale_factor)
                return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            if self.upscaler:
                try:
                    upscaled, _ = self.upscaler.enhance(frame, outscale=self.upscale_factor)
                    return upscaled
                except Exception as e:
                    print(f"AI upscaling error: {e}")
                    pass
            
            height, width = frame.shape[:2]
            new_width = int(width * self.upscale_factor)
            new_height = int(height * self.upscale_factor)
            
            upscaled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            return upscaled
            
        except Exception as e:
            print(f"Upscaling error: {e}")
            return frame
    
    def swap_face(self, frame, face_bbox):
        try:
            if not self.face_swap_enabled:
                return frame
                
            if not self.face_swapper or not self.target_face_embedding:
                if face_bbox:
                    x, y, w, h = face_bbox
                    cv2.putText(frame, "FACE SWAP: INSTALL INSIGHTFACE", (x, y-30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                return frame
            
            faces = self.face_analyzer.get(frame)
            if len(faces) == 0:
                if face_bbox:
                    x, y, w, h = face_bbox
                    cv2.putText(frame, "FACE SWAP: NO FACE DETECTED", (x, y-30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                return frame
            
            source_face = faces[0]
            
            swapped_frame = self.face_swapper.get(
                frame, 
                source_face, 
                self.target_face_embedding, 
                paste_back=True
            )
            
            if face_bbox:
                x, y, w, h = face_bbox
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                
                center = (x + w//2, y + h//2)
                axes = (int(w*0.7), int(h*0.9)) 
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                
                mask = cv2.GaussianBlur(mask, (51, 51), 0)
                mask = mask.astype(np.float32) / 255.0
                
                mask = np.maximum(mask, 0.1)
                mask = np.stack([mask] * 3, axis=-1)
                
                result = frame * (1 - mask) + swapped_frame * mask
                return result.astype(np.uint8)
            else:
                return swapped_frame
                
        except Exception as e:
            print(f"Face swap error: {e}")
            if face_bbox:
                x, y, w, h = face_bbox
                cv2.putText(frame, f"FACE SWAP ERROR: {str(e)[:20]}", (x, y-30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            return frame
    
    def interpolate_frame(self, current_frame, previous_frame=None):
        try:
            if not self.fps_boost_enabled or previous_frame is None:
                return [current_frame]
            
            if current_frame.shape != previous_frame.shape:
                previous_frame = cv2.resize(previous_frame, (current_frame.shape[1], current_frame.shape[0]))
            
            interpolated_frames = []
            
            for i in range(1, 3): 
                alpha = i / 3.0 
                
                interpolated = cv2.addWeighted(previous_frame, 1-alpha, current_frame, alpha, 0)
                
                interpolated = cv2.GaussianBlur(interpolated, (3, 3), 0.5)
                
                interpolated_frames.append(interpolated)
            
            return [previous_frame] + interpolated_frames + [current_frame]
            
        except Exception as e:
            print(f"Frame interpolation error: {e}")
            return [current_frame]
    
    def set_performance_settings(self, target_fps, quality_setting):
        self.target_fps = int(target_fps)
        self.quality_setting = quality_setting
        
        if quality_setting == "Performance":
            self.upscale_factor = 1.1
            self.denoise_strength = 0.2
            self.sharpen_strength = 0.1
        elif quality_setting == "Balanced":
            self.upscale_factor = 1.25
            self.denoise_strength = 0.3
            self.sharpen_strength = 0.15
        else:
            self.upscale_factor = 1.4
            self.denoise_strength = 0.4
            self.sharpen_strength = 0.2
    
    def should_skip_processing(self, current_fps):
        if self.quality_setting == "Performance":
            return False 
        
        if current_fps < self.target_fps * 0.8: 
            self.processing_skip_counter += 1
            if self.quality_setting == "Balanced":
                return self.processing_skip_counter % 2 == 0  
            else: 
                return self.processing_skip_counter % 3 == 0 
        
        return False
    
    def process_frame(self, frame, face_bbox=None, face_landmarks=None, previous_frame=None, current_fps=30):
        try:
            if not self.enabled:
                return [frame]
            
            if self.should_skip_processing(current_fps):
                return [frame]
            
            processed_frame = frame.copy()
            
            face_mask = None
            if face_landmarks:
                face_mask = self.create_stable_face_mask(frame, face_landmarks)
            
            if self.denoise_enabled:
                processed_frame = self.professional_denoise(processed_frame, face_mask)
            
            if self.stabilization_enabled and self.quality_setting != "Performance":
                processed_frame = self.temporal_stabilization(processed_frame)
            
            if self.quality_setting == "Performance":
                pass
            else:
                processed_frame = self.professional_upscale(processed_frame)
            
            if self.sharpen_enabled:
                processed_frame = self.intelligent_sharpen(processed_frame, face_mask)
            
            if self.fps_boost_enabled and previous_frame is not None and self.quality_setting != "Performance":
                try:
                    interpolated_frames = self.interpolate_frame(processed_frame, previous_frame)
                    self.previous_frame = processed_frame.copy()
                    return interpolated_frames
                except Exception as e:
                    print(f"FPS boost error: {e}")
            
            self.previous_frame = processed_frame.copy()
            
            cleanup_interval = 60 if self.quality_setting == "Performance" else 30
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 1
                
            if self._frame_count % cleanup_interval == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return [processed_frame]
            
        except Exception as e:
            print(f"Processing pipeline error: {e}")
            return [frame]

class VirtualCamera:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.camera = None
        self.running = False
        self.enabled = False
        self.available = self.check_availability()
        
    def check_availability(self):
        try:
            import pyvirtualcam
            return True
        except ImportError:
            return False
    
    def update_resolution(self, resolution_setting):
        if resolution_setting == "1080p":
            self.width, self.height = 1920, 1080
        elif resolution_setting == "4K":
            self.width, self.height = 3840, 2160
        elif resolution_setting == "8K":
            self.width, self.height = 7680, 4320
        
        if self.running and self.enabled:
            self.stop()
            self.start()
        
    def start(self):
        if not self.available or not self.enabled:
            return False
            
        try:
            import pyvirtualcam
            self.camera = pyvirtualcam.Camera(width=self.width, height=self.height, fps=60)
            self.running = True
            print(f"Virtual camera started: {self.width}x{self.height}")
            return True
        except Exception as e:
            print(f"Failed to start virtual camera: {e}")
            return False
    
    def send_frame(self, frame):
        if self.camera and self.running and self.enabled and self.available:
            try:
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.camera.send(frame_rgb)
            except Exception as e:
                print(f"Error sending frame: {e}")
    
    def stop(self):
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
            self.running = False

class EnhancedCameraApp:
    def __init__(self):
        cv2.setLogLevel(0)
        import os
        os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
        
        self.root = ctk.CTk()
        self.root.title("Visagyn - AI Driven Facial Engine")
        self.root.geometry("1250x1000")
        self.root.configure(fg_color=("#f0f0f0", "#1a1a1a"))
        
        self.cap = None
        self.face_tracker = FastFaceTracker()
        self.ai_processor = ProfessionalAIProcessor()
        self.virtual_cam = VirtualCamera()
        self.running = False
        self.preview_running = False
        
        self.frame_queue = queue.Queue(maxsize=3)
        self.display_queue = queue.Queue(maxsize=2)
        self.processing_queue = queue.Queue(maxsize=2)
        
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_history = deque(maxlen=30)
        self.current_fps = 30  
        
        self.setup_variables()
        self.setup_ui()
        
    def setup_variables(self):
        self.face_tracking_var = ctk.BooleanVar(value=False)
        self.ai_processing_var = ctk.BooleanVar(value=False)
        self.virtual_camera_var = ctk.BooleanVar(value=False)
        self.show_bbox_var = ctk.BooleanVar(value=False)
        self.show_mesh_var = ctk.BooleanVar(value=False)
        
        self.denoise_var = ctk.BooleanVar(value=True)
        self.sharpen_var = ctk.BooleanVar(value=True)
        self.stabilization_var = ctk.BooleanVar(value=True)
        self.face_swap_var = ctk.BooleanVar(value=False)
        self.fps_boost_var = ctk.BooleanVar(value=False)
        
        self.camera_device_var = ctk.StringVar(value="No Camera")
        self.resolution_var = ctk.StringVar(value="4K")
        self.quality_var = ctk.StringVar(value="Balanced")
        self.mesh_density_var = ctk.StringVar(value="Detailed")
        self.target_face_file = ctk.StringVar(value="No face selected")
        
        self.auto_exposure_var = ctk.BooleanVar(value=True)
        self.autofocus_var = ctk.BooleanVar(value=True)
        
        self.denoise_strength_var = ctk.DoubleVar(value=0.3)
        self.sharpen_strength_var = ctk.DoubleVar(value=0.15)
        
        self.available_cameras = []
        self.camera_connected = False
        
    def setup_ui(self):
        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.create_preview_panel()
        self.create_control_panel()
        
    def create_preview_panel(self):
        self.preview_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.preview_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.preview_label = ctk.CTkLabel(
            self.preview_frame, 
            text="Camera Preview",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.preview_label.pack(pady=20)
        
        self.video_label = ctk.CTkLabel(
            self.preview_frame,
            text="No camera connected",
            width=720,
            height=540
        )
        self.video_label.pack(padx=20, pady=20, expand=True)
        
        self.status_frame = ctk.CTkFrame(self.preview_frame, height=60, corner_radius=10)
        self.status_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        self.fps_label = ctk.CTkLabel(self.status_frame, text="FPS: 0", font=ctk.CTkFont(size=14, weight="bold"))
        self.fps_label.pack(side="left", padx=20, pady=15)
        
        self.resolution_label = ctk.CTkLabel(self.status_frame, text="Resolution: N/A", font=ctk.CTkFont(size=14))
        self.resolution_label.pack(side="left", padx=20, pady=15)
        
        self.gpu_label = ctk.CTkLabel(
            self.status_frame, 
            text=f"Processing: {'GPU' if torch.cuda.is_available() else 'CPU'}", 
            font=ctk.CTkFont(size=14)
        )
        self.gpu_label.pack(side="right", padx=20, pady=15)
        
    def create_control_panel(self):
        self.control_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.control_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        
        self.scroll_frame = ctk.CTkScrollableFrame(self.control_frame, corner_radius=0)
        self.scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        title_label = ctk.CTkLabel(
            self.scroll_frame,
            text="Controls",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=15)
        
        self.create_camera_controls()
        self.create_feature_toggles()
        self.create_ai_controls()
        self.create_performance_panel()
        self.create_action_buttons()
        
    def create_camera_controls(self):
        camera_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
        camera_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(camera_frame, text="Camera Setup", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.scan_button = ctk.CTkButton(
            camera_frame,
            text="Scan for Cameras",
            command=self.scan_cameras,
            height=32,
            font=ctk.CTkFont(size=13)
        )
        self.scan_button.pack(pady=8, padx=15, fill="x")
        
        device_frame = ctk.CTkFrame(camera_frame, fg_color="transparent")
        device_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(device_frame, text="Device:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.device_menu = ctk.CTkOptionMenu(
            device_frame,
            values=["No Camera"],
            variable=self.camera_device_var,
            width=120,
            state="disabled"
        )
        self.device_menu.pack(side="right")
        
        self.connect_button = ctk.CTkButton(
            camera_frame,
            text="Connect Camera",
            command=self.connect_camera,
            height=32,
            font=ctk.CTkFont(size=13),
            state="disabled"
        )
        self.connect_button.pack(pady=8, padx=15, fill="x")
        
        try:
            face_swap_status = "Not Available"
            if hasattr(self.face_tracker, 'face_swapping_available'):
                if self.face_tracker.face_swapping_available:
                    face_swap_status = "Available"
                    
            print(f"Face Swapping Status: {face_swap_status}")
            print(f"Face Tracker has face_analyzer: {hasattr(self.face_tracker, 'face_analyzer')}")
            print(f"Face Tracker has face_swapper: {hasattr(self.face_tracker, 'face_swapper')}")
        except Exception as e:
            print(f"Debug error: {e}")
        
        resolution_frame = ctk.CTkFrame(camera_frame, fg_color="transparent")
        resolution_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(resolution_frame, text="Output:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.resolution_menu = ctk.CTkOptionMenu(
            resolution_frame,
            values=["1080p", "4K", "8K"],
            variable=self.resolution_var,
            width=80
        )
        self.resolution_menu.pack(side="right")
        
        controls_separator = ctk.CTkFrame(camera_frame, height=2, fg_color="gray")
        controls_separator.pack(fill="x", padx=15, pady=8)
        
        ctk.CTkLabel(camera_frame, text="Camera Controls", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5, 0))
        
        self.auto_exposure_toggle = ctk.CTkSwitch(
            camera_frame,
            text="Auto Exposure",
            variable=self.auto_exposure_var,
            command=self.update_camera_settings,
            font=ctk.CTkFont(size=12),
            state="disabled"
        )
        self.auto_exposure_toggle.pack(pady=3, padx=20, anchor="w")
        
        self.autofocus_toggle = ctk.CTkSwitch(
            camera_frame,
            text="Auto Focus",
            variable=self.autofocus_var,
            command=self.update_camera_settings,
            font=ctk.CTkFont(size=12),
            state="disabled"
        )
        self.autofocus_toggle.pack(pady=3, padx=20, anchor="w")
        
    def create_feature_toggles(self):
        features_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
        features_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(features_frame, text="Detection Features", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.face_toggle = ctk.CTkSwitch(
            features_frame,
            text="Face Tracking",
            variable=self.face_tracking_var,
            command=self.toggle_face_tracking,
            font=ctk.CTkFont(size=13),
            state="disabled"
        )
        self.face_toggle.pack(pady=5, padx=20, anchor="w")
        
        mesh_text = "Face Mesh Overlay" if self.face_tracker.use_mediapipe else "Face Mesh (Needs MediaPipe)"
        self.mesh_toggle = ctk.CTkSwitch(
            features_frame,
            text=mesh_text,
            variable=self.show_mesh_var,
            font=ctk.CTkFont(size=13),
            state="disabled"
        )
        self.mesh_toggle.pack(pady=5, padx=20, anchor="w")
        
        self.bbox_toggle = ctk.CTkSwitch(
            features_frame,
            text="Face Bounding Box",
            variable=self.show_bbox_var,
            font=ctk.CTkFont(size=13),
            state="disabled"
        )
        self.bbox_toggle.pack(pady=5, padx=20, anchor="w")
        
        face_swap_separator = ctk.CTkFrame(features_frame, height=2, fg_color="gray")
        face_swap_separator.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(features_frame, text="DeepFake", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5, 0))
        
        face_swap_available = (hasattr(self.face_tracker, 'face_analyzer') and 
                              self.face_tracker.face_analyzer is not None and
                              hasattr(self.face_tracker, 'face_swapper') and
                              self.face_tracker.face_swapper is not None)
        
        if face_swap_available:
            face_swap_text = "Enable Face Swap"
        elif (hasattr(self.face_tracker, 'face_analyzer') and 
              self.face_tracker.face_analyzer is not None):
            face_swap_text = "Face Swap (Err)"
        else:
            face_swap_text = "Face Swap (See Csl)"
        
        self.face_swap_toggle = ctk.CTkSwitch(
            features_frame,
            text=face_swap_text,
            variable=self.face_swap_var,
            command=self.toggle_face_swap,
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled"
        )
        self.face_swap_toggle.pack(pady=5, padx=20, anchor="w")
        
        self.face_select_frame = ctk.CTkFrame(features_frame, fg_color="transparent")
        
        self.select_face_button = ctk.CTkButton(
            self.face_select_frame,
            text="Select Target Face",
            command=self.select_target_face,
            height=28,
            width=120,
            font=ctk.CTkFont(size=11),
            state="disabled"
        )
        self.select_face_button.pack(side="left", padx=(20, 8), pady=2)
        
        self.face_status_label = ctk.CTkLabel(
            self.face_select_frame,
            textvariable=self.target_face_file,
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.face_status_label.pack(side="left", pady=2)
        
        self.face_select_frame.pack(fill="x", pady=2)
        
        if face_swap_available:
            info_text = "Uses AI + Face Mesh for swapping"
            info_color = "cyan"
        elif (hasattr(self.face_tracker, 'face_analyzer') and 
              self.face_tracker.face_analyzer is not None):
            info_text = "Model download failed - check console!"
            info_color = "orange"
        else:
            info_text = "Install: pip install insightface onnxruntime"
            info_color = "orange"
            
        face_swap_info = ctk.CTkLabel(
            features_frame,
            text=info_text,
            font=ctk.CTkFont(size=9),
            text_color=info_color
        )
        face_swap_info.pack(pady=(0, 5), padx=20, anchor="w")
        
    def create_performance_panel(self):
        perf_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
        perf_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(perf_frame, text="Performance", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        fps_frame = ctk.CTkFrame(perf_frame, fg_color="transparent")
        fps_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(fps_frame, text="Target FPS:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.fps_target = ctk.CTkOptionMenu(
            fps_frame,
            values=["30", "60", "120"],
            command=self.update_fps_target,
            width=80
        )
        self.fps_target.pack(side="right")
        self.fps_target.set("60")
        
        quality_frame = ctk.CTkFrame(perf_frame, fg_color="transparent")
        quality_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(quality_frame, text="Quality:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.quality_menu = ctk.CTkOptionMenu(
            quality_frame,
            values=["Performance", "Balanced", "Quality"],
            variable=self.quality_var,
            command=self.update_quality_preset,
            width=100
        )
        self.quality_menu.pack(side="right")
        
        mesh_frame = ctk.CTkFrame(perf_frame, fg_color="transparent")
        mesh_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(mesh_frame, text="Mesh Detail:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.mesh_detail = ctk.CTkOptionMenu(
            mesh_frame,
            values=["Simple", "Detailed", "Ultra"],
            variable=self.mesh_density_var,
            width=100
        )
        self.mesh_detail.pack(side="right")
        self.mesh_detail.set("Detailed")
        
    def create_ai_controls(self):
        ai_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
        ai_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(ai_frame, text="AI Enhancement", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.ai_toggle = ctk.CTkSwitch(
            ai_frame,
            text="Enable AI Processing",
            variable=self.ai_processing_var,
            command=self.toggle_ai_processing,
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled"
        )
        self.ai_toggle.pack(pady=5, padx=20, anchor="w")
        
        self.ai_options_frame = ctk.CTkFrame(ai_frame, fg_color="transparent")
        
        self.denoise_toggle = ctk.CTkSwitch(
            self.ai_options_frame,
            text="Smart Denoising",
            variable=self.denoise_var,
            command=self.toggle_denoise,
            font=ctk.CTkFont(size=12)
        )
        self.denoise_toggle.pack(pady=3, padx=30, anchor="w")
        
        denoise_frame = ctk.CTkFrame(self.ai_options_frame, fg_color="transparent")
        denoise_frame.pack(fill="x", padx=40, pady=2)
        ctk.CTkLabel(denoise_frame, text="Strength:", font=ctk.CTkFont(size=10)).pack(side="left")
        self.denoise_slider = ctk.CTkSlider(
            denoise_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.denoise_strength_var,
            command=self.update_denoise_strength,
            width=80
        )
        self.denoise_slider.pack(side="right")
        
        self.sharpen_toggle = ctk.CTkSwitch(
            self.ai_options_frame,
            text="Intelligent Sharpening",
            variable=self.sharpen_var,
            command=self.toggle_sharpen,
            font=ctk.CTkFont(size=12)
        )
        self.sharpen_toggle.pack(pady=3, padx=30, anchor="w")
        
        sharpen_frame = ctk.CTkFrame(self.ai_options_frame, fg_color="transparent")
        sharpen_frame.pack(fill="x", padx=40, pady=2)
        ctk.CTkLabel(sharpen_frame, text="Strength:", font=ctk.CTkFont(size=10)).pack(side="left")
        self.sharpen_slider = ctk.CTkSlider(
            sharpen_frame, 
            from_=0.05, 
            to=0.5, 
            variable=self.sharpen_strength_var,
            command=self.update_sharpen_strength,
            width=80
        )
        self.sharpen_slider.pack(side="right")
        
        self.stabilization_toggle = ctk.CTkSwitch(
            self.ai_options_frame,
            text="Temporal Stabilization",
            variable=self.stabilization_var,
            command=self.toggle_stabilization,
            font=ctk.CTkFont(size=12)
        )
        self.stabilization_toggle.pack(pady=3, padx=30, anchor="w")
        
        self.fps_boost_toggle = ctk.CTkSwitch(
            self.ai_options_frame,
            text="FPS Boost",
            variable=self.fps_boost_var,
            command=self.toggle_fps_boost,
            font=ctk.CTkFont(size=12)
        )
        self.fps_boost_toggle.pack(pady=3, padx=30, anchor="w")
        
        quality_frame = ctk.CTkFrame(self.ai_options_frame, fg_color="transparent")
        quality_frame.pack(fill="x", padx=30, pady=5)
        ctk.CTkLabel(quality_frame, text="Quality:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.quality_menu = ctk.CTkOptionMenu(
            quality_frame,
            values=["Performance", "Balanced", "Quality"],
            variable=self.quality_var,
            command=self.update_quality_preset,
            width=100
        )
        self.quality_menu.pack(side="right")
        
        vcam_text = "Virtual Camera Output" if self.virtual_cam.available else "Virtual Camera (Not Available)"
        self.vcam_toggle = ctk.CTkSwitch(
            ai_frame,
            text=vcam_text,
            variable=self.virtual_camera_var,
            command=self.toggle_virtual_camera,
            font=ctk.CTkFont(size=13),
            state="disabled"
        )
        self.vcam_toggle.pack(pady=5, padx=20, anchor="w")
        
    def create_action_buttons(self):
        button_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=15, pady=15)
        
        self.start_button = ctk.CTkButton(
            button_frame,
            text="Start Processing",
            command=self.toggle_processing,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=10,
            state="disabled"
        )
        self.start_button.pack(fill="x", pady=5)
        
        info_label = ctk.CTkLabel(
            button_frame,
            text="💡 Tip: Use 'Performance' mode for 60+ FPS. 'Quality' mode for best results.",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        info_label.pack(pady=5)
    
    def scan_cameras(self):
        self.scan_button.configure(text="Scanning...", state="disabled")
        self.available_cameras = []
        
        def scan_thread():
            cameras_found = []
            
            print("Scanning for cameras...")
            
            import os
            import sys
            from io import StringIO
            
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            
            try:
                backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
                
                for backend in backends:
                    for i in range(5):
                        try:
                            print(f"Testing camera {i} with backend {backend}")
                            cap = cv2.VideoCapture(i, backend)
                            
                            if cap.isOpened():
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                
                                ret, frame = cap.read()
                                if ret and frame is not None and frame.size > 0:
                                    camera_name = f"Camera {i}"
                                    if camera_name not in cameras_found:
                                        cameras_found.append(camera_name)
                                        print(f"Found working camera: {camera_name}")
                                
                                cap.release()
                                time.sleep(0.1)
                            else:
                                cap.release()
                                
                        except Exception as e:
                            print(f"Error testing camera {i}: {e}")
                            continue
                    
                    if cameras_found:
                        break
                        
            finally:
                sys.stderr = old_stderr
                        
            self.available_cameras = cameras_found
            print(f"Scan complete. Found cameras: {cameras_found}")
            self.root.after(0, self.update_camera_list)
            
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def update_camera_list(self):
        if self.available_cameras:
            self.device_menu.configure(values=self.available_cameras, state="normal")
            self.device_menu.set(self.available_cameras[0])
            self.connect_button.configure(state="normal")
            self.scan_button.configure(text="Rescan Cameras", state="normal")
            print(f"Camera list updated: {self.available_cameras}")
        else:
            self.device_menu.configure(values=["No Camera Found"], state="disabled")
            self.device_menu.set("No Camera Found")
            self.connect_button.configure(state="disabled")
            self.scan_button.configure(text="Rescan Cameras", state="normal")
            messagebox.showwarning("No Cameras", "No cameras detected. Make sure your camera is connected and not being used by another application.")
            print("No cameras found during scan")
    
    def connect_camera(self):
        if not self.available_cameras:
            messagebox.showerror("Error", "No cameras available.")
            return
            
        selected = self.camera_device_var.get()
        if selected == "No Camera Found":
            return
            
        camera_id = int(selected.split()[-1])
        
        if self.initialize_camera_connection(camera_id):
            self.camera_connected = True
            self.connect_button.configure(text="Disconnect", command=self.disconnect_camera)
            self.start_button.configure(state="normal")
            self.enable_controls()
            
            self.start_preview()
            messagebox.showinfo("Success", f"Connected to {selected} - Preview active")
        else:
            messagebox.showerror("Error", f"Failed to connect to {selected}")
    
    def disconnect_camera(self):
        self.running = False
        self.preview_running = False
        time.sleep(0.2)
            
        self.camera_connected = False
        self.connect_button.configure(text="Connect Camera", command=self.connect_camera)
        self.start_button.configure(text="Start Processing", state="disabled")
        self.disable_controls()
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.video_label.configure(text="Camera disconnected")
        if hasattr(self.video_label, 'image'):
            self.video_label.configure(image="")
            self.video_label.image = None
        
        self.resolution_label.configure(text="Resolution: N/A")
        self.fps_label.configure(text="FPS: 0")
        
    def enable_controls(self):
        self.face_toggle.configure(state="normal")
        self.ai_toggle.configure(state="normal")
        
        if self.face_tracker.use_mediapipe:
            self.mesh_toggle.configure(state="normal")
            
        if self.virtual_cam.available:
            self.vcam_toggle.configure(state="normal")
            
        self.bbox_toggle.configure(state="normal")
        
        if (hasattr(self.face_tracker, 'face_analyzer') and 
            self.face_tracker.face_analyzer is not None and
            hasattr(self.face_tracker, 'face_swapper') and
            self.face_tracker.face_swapper is not None):
            self.face_swap_toggle.configure(state="normal")
            self.select_face_button.configure(state="normal")
            print("Face swap controls enabled - full functionality")
        elif (hasattr(self.face_tracker, 'face_analyzer') and 
              self.face_tracker.face_analyzer is not None):
            self.face_swap_toggle.configure(state="normal")
            self.select_face_button.configure(state="disabled")
            print("Face swap controls partially enabled - model download failed")
        else:
            self.face_swap_toggle.configure(state="disabled")
            self.select_face_button.configure(state="disabled")
            print("Face swap controls disabled - InsightFace not available")
        
        self.auto_exposure_toggle.configure(state="normal")
        self.autofocus_toggle.configure(state="normal")
        
    def disable_controls(self):
        controls = [self.face_toggle, self.mesh_toggle, self.bbox_toggle, 
                   self.ai_toggle, self.vcam_toggle, self.auto_exposure_toggle, 
                   self.autofocus_toggle, self.face_swap_toggle, self.select_face_button]
        for control in controls:
            control.configure(state="disabled")
        
        self.face_tracking_var.set(False)
        self.show_mesh_var.set(False)
        self.show_bbox_var.set(False)
        self.ai_processing_var.set(False)
        self.virtual_camera_var.set(False)
        self.face_swap_var.set(False)
        
        if hasattr(self.face_tracker, 'face_swap_enabled'):
            self.face_tracker.face_swap_enabled = False
    
    def toggle_face_tracking(self):
        self.face_tracker.enabled = self.face_tracking_var.get()
        
    def toggle_face_swap(self):
        if not hasattr(self.face_tracker, 'face_analyzer') or not self.face_tracker.face_analyzer:
            messagebox.showwarning(
                "InsightFace Required", 
                "Face swapping requires InsightFace.\n\n"
                "Install with:\n"
                "pip install insightface onnxruntime\n\n"
                "Note: May require Visual Studio C++ Build Tools on Windows."
            )
            self.face_swap_var.set(False)
            return
            
        if not hasattr(self.face_tracker, 'face_swapper') or not self.face_tracker.face_swapper:
            user_path = os.path.expanduser("~")
            model_dir = os.path.join(user_path, ".insightface", "models")
            messagebox.showerror(
                "Face Swapper Model Missing", 
                "Face swapping model download failed.\n\n"
                "Manual fix:\n"
                "1. Download inswapper_128.onnx from:\n"
                "   https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx\n\n"
                f"2. Place it in:\n   {model_dir}\n\n"
                "3. Restart the application"
            )
            self.face_swap_var.set(False)
            return
        
        self.face_tracker.face_swap_enabled = self.face_swap_var.get()
        if (self.face_swap_var.get() and 
            hasattr(self.face_tracker, 'target_face_embedding') and 
            not self.face_tracker.target_face_embedding):
            messagebox.showwarning("No Target Face", "Please select a target face image first.")
            self.face_swap_var.set(False)
            self.face_tracker.face_swap_enabled = False
    
    def select_target_face(self):
        print("Attempting to select target face...")
        
        if not hasattr(self.face_tracker, 'face_analyzer') or not self.face_tracker.face_analyzer:
            print("Face analyzer not available")
            messagebox.showerror(
                "InsightFace Required", 
                "Face swapping requires InsightFace.\n\n"
                "Install with:\n"
                "pip install insightface onnxruntime\n\n"
                "Note: May require Visual Studio C++ Build Tools on Windows."
            )
            return
        
        print("Face analyzer available, opening file dialog...")
        
        file_path = filedialog.askopenfilename(
            title="Select Target Face Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            print(f"Selected file: {file_path}")
            try:
                if self.face_tracker.set_target_face(file_path):
                    filename = os.path.basename(file_path)
                    self.target_face_file.set(f"✓ {filename[:15]}...")
                    print(f"Successfully loaded target face: {filename}")
                    messagebox.showinfo("Success", f"Target face loaded: {filename}")
                else:
                    print("Failed to load target face")
                    messagebox.showerror("Error", "Failed to load target face. Check console for details.\n\nTips:\n• Use clear, well-lit photos\n• Face should be facing forward\n• Avoid glasses or face coverings")
            except Exception as e:
                print(f"Exception during face loading: {e}")
                messagebox.showerror("Error", f"Error loading face: {str(e)}")
        else:
            print("No file selected")
        
    def toggle_ai_processing(self):
        self.ai_processor.enabled = self.ai_processing_var.get()
        
        if self.ai_processing_var.get():
            self.ai_options_frame.pack(fill="x", pady=5)
        else:
            self.ai_options_frame.pack_forget()
            self.fps_boost_var.set(False)
            self.ai_processor.fps_boost_enabled = False
    
    def toggle_denoise(self):
        self.ai_processor.denoise_enabled = self.denoise_var.get()
        
    def toggle_sharpen(self):
        self.ai_processor.sharpen_enabled = self.sharpen_var.get()
        
    def toggle_stabilization(self):
        self.ai_processor.stabilization_enabled = self.stabilization_var.get()
        
    def toggle_fps_boost(self):
        self.ai_processor.fps_boost_enabled = self.fps_boost_var.get()
        
    def toggle_face_swap(self):
        self.ai_processor.face_swap_enabled = self.face_swap_var.get()
        if self.face_swap_var.get() and not self.face_tracker.target_face_embedding:
            messagebox.showwarning("No Target Face", "Please select a target face image first.")
            self.face_swap_var.set(False)
            self.ai_processor.face_swap_enabled = False
    
    def update_denoise_strength(self, value):
        self.ai_processor.denoise_strength = float(value)
        
    def update_sharpen_strength(self, value):
        self.ai_processor.sharpen_strength = float(value)
        
    def update_quality_preset(self, preset):
        if preset == "Performance":
            self.ai_processor.upscale_factor = 1.1
            self.denoise_strength_var.set(0.2)
            self.sharpen_strength_var.set(0.1)
        elif preset == "Balanced":
            self.ai_processor.upscale_factor = 1.25
            self.denoise_strength_var.set(0.3)
            self.sharpen_strength_var.set(0.15)
        else: 
            self.ai_processor.upscale_factor = 1.5
            self.denoise_strength_var.set(0.4)
            self.sharpen_strength_var.set(0.2)
            
        self.ai_processor.denoise_strength = self.denoise_strength_var.get()
        self.ai_processor.sharpen_strength = self.sharpen_strength_var.get()
        
        target_fps = int(self.fps_target.get()) if hasattr(self, 'fps_target') else 60
        self.ai_processor.set_performance_settings(target_fps, preset)
        
        print(f"Quality preset changed to: {preset}")
    
    def update_fps_target(self, fps_target):
        target_fps = int(fps_target)
        quality_setting = self.quality_var.get()
        self.ai_processor.set_performance_settings(target_fps, quality_setting)
        print(f"FPS target changed to: {fps_target}")
    
    def select_target_face(self):
        file_path = filedialog.askopenfilename(
            title="Select Target Face Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            if self.ai_processor.set_target_face(file_path):
                filename = os.path.basename(file_path)
                self.target_face_file.set(f"✓ {filename[:15]}...")
                messagebox.showinfo("Success", f"Target face loaded: {filename}")
            else:
                messagebox.showerror("Error", "Failed to load target face.")
    
    def toggle_virtual_camera(self):
        if not self.virtual_cam.available:
            messagebox.showwarning("Virtual Camera Unavailable", 
                                 "Install with: pip install pyvirtualcam")
            self.virtual_camera_var.set(False)
            return
            
        self.virtual_cam.enabled = self.virtual_camera_var.get()
        
        self.virtual_cam.update_resolution(self.resolution_var.get())
        
        if self.virtual_cam.enabled and not self.virtual_cam.running:
            if not self.virtual_cam.start():
                messagebox.showerror("Error", "Failed to start virtual camera.")
                self.virtual_camera_var.set(False)
                self.virtual_cam.enabled = False
            else:
                messagebox.showinfo("Virtual Camera", f"Started at {self.resolution_var.get()} resolution")
        elif not self.virtual_cam.enabled and self.virtual_cam.running:
            self.virtual_cam.stop()
    
    def update_camera_settings(self):
        if self.cap and self.cap.isOpened():
            try:
                if self.auto_exposure_var.get():
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                else:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
                
                if self.autofocus_var.get():
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1) 
                else:
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
                
                print(f"Camera settings updated: Auto Exposure: {self.auto_exposure_var.get()}, Auto Focus: {self.autofocus_var.get()}")
            except Exception as e:
                print(f"Error updating camera settings: {e}")
    
    def initialize_camera_connection(self, camera_id):
        try:
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_id)
                if not self.cap.isOpened():
                    return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 60) 
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            self.update_camera_settings()
            
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.cap.release()
                return False
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.root.after(0, lambda: self.resolution_label.configure(
                text=f"Input: {actual_width}x{actual_height}@{actual_fps}fps"
            ))
            
            print(f"Camera initialized: {actual_width}x{actual_height}@{actual_fps}fps")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def start_preview(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        self.preview_running = True
        self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
        self.preview_thread.start()
        
    def preview_loop(self):
        while self.preview_running and self.camera_connected:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.update_preview(frame)
                    self.update_fps_counter()
                    time.sleep(1/30)  
                else:
                    break
            except Exception as e:
                print(f"Preview error: {e}")
                break
    
    def start_threads(self):
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        
        self.capture_thread.start()
        self.processing_thread.start()
        self.display_thread.start()
    
    def capture_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    try:
                        self.frame_queue.put(frame, timeout=0.001)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame, timeout=0.001)
                        except:
                            pass
            except Exception as e:
                print(f"Capture error: {e}")
                
    def processing_loop(self):
        frame_skip_counter = 0
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                skip_ai = False
                if len(self.fps_history) > 10:
                    avg_fps = sum(self.fps_history) / len(self.fps_history)
                    if avg_fps < 25 and self.ai_processing_var.get():
                        frame_skip_counter += 1
                        if frame_skip_counter % 2 == 0:
                            skip_ai = True
                
                if skip_ai:
                    processed_frame = frame 
                else:
                    processed_frame = self.process_frame_fast(frame)
                
                try:
                    self.display_queue.put(processed_frame, timeout=0.001)
                except queue.Full:
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put(processed_frame, timeout=0.001)
                    except:
                        pass
                
                if self.virtual_cam.enabled:
                    try:
                        output_resolution = self.resolution_var.get()
                        if output_resolution == "4K":
                            output_frame = cv2.resize(processed_frame, (3840, 2160), interpolation=cv2.INTER_LINEAR)
                        elif output_resolution == "8K":
                            output_frame = cv2.resize(processed_frame, (7680, 4320), interpolation=cv2.INTER_LINEAR)
                        else:  
                            output_frame = cv2.resize(processed_frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                            
                        self.virtual_cam.send_frame(output_frame)
                    except Exception as e:
                        print(f"Virtual camera error: {e}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def display_loop(self):
        while self.running:
            try:
                frame = self.display_queue.get(timeout=1.0)
                self.update_preview_fast(frame)
                self.update_fps_counter()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Display error: {e}")
    
    def process_frame_fast(self, frame):
        try:
            display_frame = frame.copy()
            face_bbox = None
            face_landmarks = None
            
            if self.face_tracking_var.get():
                face_bbox = self.face_tracker.detect_face_fast(frame)
                
                if self.face_tracker.use_mediapipe:
                    landmarks, _ = self.face_tracker.get_face_mesh(frame)
                    if landmarks:
                        face_landmarks = landmarks
                        
                        if (self.face_swap_var.get() and 
                            hasattr(self.face_tracker, 'face_swap_enabled') and 
                            self.face_tracker.face_swap_enabled and
                            hasattr(self.face_tracker, 'swap_face_with_mesh')):
                            try:
                                display_frame = self.face_tracker.swap_face_with_mesh(display_frame, landmarks)
                                
                                if (hasattr(self.face_tracker, 'target_face_embedding') and 
                                    self.face_tracker.target_face_embedding):
                                    cv2.putText(display_frame, "FACE SWAP ACTIVE", (10, 60), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                                else:
                                    cv2.putText(display_frame, "FACE SWAP: NO TARGET", (10, 60), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            except Exception as e:
                                print(f"Face swap processing error: {e}")
                                cv2.putText(display_frame, "FACE SWAP ERROR", (10, 60), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        if self.show_mesh_var.get():
                            self.draw_face_mesh(display_frame, landmarks)
                
                if face_bbox and self.show_bbox_var.get():
                    x, y, w, h = face_bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "FACE", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.ai_processing_var.get():
                try:
                    processed_frames = self.ai_processor.process_frame(
                        display_frame, 
                        face_bbox, 
                        face_landmarks,
                        getattr(self.ai_processor, 'previous_frame', None),
                        self.current_fps
                    )
                    
                    if processed_frames and len(processed_frames) > 0:
                        display_frame = processed_frames[-1]
                    
                    status_items = []
                    if self.ai_processor.denoise_enabled:
                        status_items.append("DENOISED")
                    if self.ai_processor.sharpen_enabled:
                        status_items.append("SHARPENED")
                    if self.ai_processor.stabilization_enabled:
                        status_items.append("STABILIZED")
                    if self.ai_processor.fps_boost_enabled:
                        status_items.append("FPS BOOST")
                        
                    if status_items:
                        status_text = " | ".join(status_items)
                        cv2.putText(display_frame, f"AI: {status_text}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                except Exception as e:
                    print(f"AI processing error: {e}")
                    cv2.putText(display_frame, "AI ERROR", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return display_frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame
    
    def update_preview_fast(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (720, 540), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray(frame_resized)
            
            ctk_image = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=(720, 540)
            )
            
            def update_ui():
                self.video_label.configure(image=ctk_image, text="")
                self.video_label.image = ctk_image
            
            self.root.after_idle(update_ui)
            
        except Exception as e:
            print(f"Preview update error: {e}")
    
    def process_frame(self, frame):
        try:
            display_frame = frame.copy()
            face_bbox = None
            face_landmarks = None
            
            if self.face_tracking_var.get():
                face_bbox = self.face_tracker.detect_face_fast(frame)
                
                if self.face_tracker.use_mediapipe:
                    landmarks, _ = self.face_tracker.get_face_mesh(frame)
                    if landmarks:
                        face_landmarks = landmarks
                        
                        if self.show_mesh_var.get():
                            self.draw_face_mesh(display_frame, landmarks)
                
                if face_bbox and self.show_bbox_var.get():
                    x, y, w, h = face_bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "FACE", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.ai_processing_var.get():
                try:
                    processed_frames = self.ai_processor.process_frame(
                        display_frame, 
                        face_bbox, 
                        face_landmarks,
                        getattr(self.ai_processor, 'previous_frame', None),
                        self.current_fps  
                    )
                    
                    if processed_frames and len(processed_frames) > 0:
                        display_frame = processed_frames[-1] 
                    
                    status_items = []
                    if self.ai_processor.denoise_enabled:
                        status_items.append("DENOISED")
                    if self.ai_processor.sharpen_enabled:
                        status_items.append("SHARPENED")
                    if self.ai_processor.stabilization_enabled:
                        status_items.append("STABILIZED")
                    if self.ai_processor.fps_boost_enabled:
                        status_items.append("FPS BOOST")
                    if self.ai_processor.face_swap_enabled:
                        status_items.append("FACE SWAPPED")
                        
                    if status_items:
                        status_text = " | ".join(status_items)
                        cv2.putText(display_frame, f"AI: {status_text}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                except Exception as e:
                    print(f"AI processing error: {e}")
                    cv2.putText(display_frame, "AI ERROR", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return display_frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame
    
    def draw_face_mesh(self, frame, landmarks):
        try:
            import mediapipe as mp
            h, w, _ = frame.shape
            density = self.mesh_density_var.get()
            
            points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))
            
            mp_face_mesh = mp.solutions.face_mesh
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            if density == "Simple":
                connections = mp_face_mesh.FACEMESH_CONTOURS
                self.draw_mediapipe_connections(frame, points, connections, (0, 255, 255), 1)
                
            elif density == "Detailed":
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_CONTOURS, (0, 255, 255), 2)
                
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_LEFT_EYE, (255, 0, 255), 2)
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_RIGHT_EYE, (255, 0, 255), 2)
                
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_LEFT_EYEBROW, (0, 255, 0), 2)
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_RIGHT_EYEBROW, (0, 255, 0), 2)
                
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_LIPS, (0, 0, 255), 2)
                
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_FACE_OVAL, (255, 255, 0), 1)
                
            else:
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_FACE_OVAL, (255, 255, 0), 2)
                
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_LEFT_EYE, (255, 0, 255), 2)
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_RIGHT_EYE, (255, 0, 255), 2)
                
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_LEFT_EYEBROW, (0, 255, 0), 2)
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_RIGHT_EYEBROW, (0, 255, 0), 2)
                
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_LIPS, (0, 0, 255), 2)
                
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_LEFT_IRIS, (0, 255, 0), 2)
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_RIGHT_IRIS, (0, 255, 0), 2)
                
                self.draw_mediapipe_connections(frame, points, mp_face_mesh.FACEMESH_TESSELATION, (0, 255, 255), 1)
                
                key_points = [
                    1, 2, 5, 4, 6, 19, 94, 125, 
                    33, 133, 362, 263,  
                    61, 291, 39, 269, 
                    10, 151, 9, 8,  
                    175, 18, 200, 199, 
                    468, 469, 470, 471, 472, 473 
                ]
                
                for point_idx in key_points:
                    if point_idx < len(points):
                        cv2.circle(frame, points[point_idx], 3, (255, 255, 255), -1)
                        cv2.circle(frame, points[point_idx], 5, (0, 0, 0), 1)  # Black outline
                        
        except Exception as e:
            print(f"MediaPipe mesh drawing error: {e}")
            try:
                h, w, _ = frame.shape
                if landmarks and hasattr(landmarks, 'landmark') and len(landmarks.landmark) > 0:
                    for i, landmark in enumerate(landmarks.landmark):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        if 0 <= x < w and 0 <= y < h:
                            if i < 17: 
                                color = (0, 255, 255)  
                            elif i < 68: 
                                color = (255, 0, 255)  
                            elif i < 97:  
                                color = (255, 255, 0)  
                            elif i < 136: 
                                color = (0, 0, 255) 
                            else: 
                                color = (100, 100, 255)  
                            
                            size = 2 if i % 10 == 0 else 1
                            cv2.circle(frame, (x, y), size, color, -1)
            except Exception:
                pass
    
    def draw_mediapipe_connections(self, frame, points, connections, color, thickness):
        try:
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if start_idx < len(points) and end_idx < len(points):
                    start_point = points[start_idx]
                    end_point = points[end_idx]
                    cv2.line(frame, start_point, end_point, color, thickness)
        except Exception as e:
            print(f"Connection drawing error: {e}")
    
    def draw_connections(self, frame, points, indices, color, thickness, closed=False):
        try:
            for i in range(len(indices) - 1):
                if indices[i] < len(points) and indices[i + 1] < len(points):
                    cv2.line(frame, points[indices[i]], points[indices[i + 1]], color, thickness)
            
            if closed and len(indices) > 2:
                if indices[0] < len(points) and indices[-1] < len(points):
                    cv2.line(frame, points[indices[-1]], points[indices[0]], color, thickness)
        except Exception as e:
            print(f"Connection drawing error: {e}")
    
    def update_preview(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (720, 540), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray(frame_resized)
            
            ctk_image = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=(720, 540)
            )
            
            def update_ui():
                self.video_label.configure(image=ctk_image, text="")
                self.video_label.image = ctk_image
            
            self.root.after_idle(update_ui)
            
        except Exception as e:
            print(f"Preview update error: {e}")
    
    def update_fps_counter(self):
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_history.append(fps)
            self.current_fps = fps  
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            color = "green" if fps >= 25 else "orange" if fps >= 15 else "red"
            
            def update_fps_label():
                self.fps_label.configure(
                    text=f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})",
                    text_color=color
                )
            
            self.root.after_idle(update_fps_label)
            
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def toggle_processing(self):
        if not self.running:
            if not self.camera_connected or not self.cap:
                messagebox.showerror("Error", "No camera connected")
                return
            
            self.preview_running = False
            time.sleep(0.1)
            
            self.running = True
            self.start_button.configure(text="Stop Processing")
            self.start_threads()
            print("AI processing started")
            
        else:
            self.running = False
            self.start_button.configure(text="Start Processing")
            time.sleep(0.2)
            
            self.virtual_cam.stop()
            
            if self.camera_connected and self.cap:
                self.start_preview()
            else:
                self.video_label.configure(text="Processing stopped")
                if hasattr(self.video_label, 'image'):
                    self.video_label.configure(image="")
                    self.video_label.image = None
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        self.running = False
        self.preview_running = False
        time.sleep(0.3)
        
        if self.cap:
            self.cap.release()
        
        self.virtual_cam.stop()
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
            
        self.root.destroy()

def install_dependencies():
    required_modules = [
        ('customtkinter', 'customtkinter'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('PIL', 'pillow'),
        ('cv2', 'opencv-python')
    ]
    
    missing_required = []
    for module_name, pip_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing_required.append(pip_name)
    
    if missing_required:
        print(f"Missing required dependencies: {', '.join(missing_required)}")
        print(f"Install with: pip install {' '.join(missing_required)}")
        sys.exit(1)
    
    print("=" * 70)
    print("Visagyn - AI Driven Facial Engine")
    print("=" * 70)
    
    optional_deps = [
        ('mediapipe', 'MediaPipe face mesh tracking'),
        ('pyvirtualcam', 'Virtual camera support'),
        ('realesrgan', 'AI upscaling (RealESRGAN)'),
        ('insightface', 'Face swapping capabilities (integrated with face tracking)'),
    ]
    
    for module, description in optional_deps:
        try:
            __import__(module)
            print(f"✓ {description} available")
        except ImportError:
            print(f"✗ {description} not available")
    
    print("=" * 70)
    print("Fixes:")
    print("   • Improved memory management")
    print("=" * 70)

if __name__ == "__main__":
    install_dependencies()
    
    app = EnhancedCameraApp()
    app.run()