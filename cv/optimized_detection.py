# cv/optimized_detection.py
import torch
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import time

class OptimizedDetector:
    def __init__(self, model_path='yolov8s.pt', use_gpu=True, use_onnx=False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_onnx = use_onnx
        self.input_size = 640
        
        if use_onnx:
            self._setup_onnx(model_path)
        else:
            self._setup_pytorch(model_path)
        
        print(f"Detector initialized - GPU: {self.use_gpu}, ONNX: {self.use_onnx}")
    
    def _setup_pytorch(self, model_path):
        """Setup PyTorch model"""
        self.model = YOLO(model_path)
        if self.use_gpu:
            self.model.to('cuda')
        self.class_names = self.model.names
    
    def _setup_onnx(self, model_path):
        """Setup ONNX model"""
        onnx_path = model_path.replace('.pt', '.onnx')
        
        # Convert to ONNX if needed
        if not Path(onnx_path).exists():
            print(f"Converting {model_path} to ONNX...")
            temp_model = YOLO(model_path)
            temp_model.export(format='onnx', imgsz=self.input_size)
            print(f"ONNX model saved: {onnx_path}")
        
        # Setup ONNX Runtime
        providers = []
        if self.use_gpu:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Load class names from original model
        temp_model = YOLO(model_path)
        self.class_names = temp_model.names
    
    def preprocess_frame(self, frame):
        """Preprocess single frame for inference"""
        # Resize while maintaining aspect ratio
        height, width = frame.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        new_width, new_height = int(width * scale), int(height * scale)
        
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Pad to square
        pad_x = self.input_size - new_width
        pad_y = self.input_size - new_height
        padded = cv2.copyMakeBorder(resized, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=114)
        
        # Normalize and convert to tensor format
        image = padded.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        return image, scale, (pad_x, pad_y)
    
    def preprocess_batch(self, frames):
        """Preprocess batch of frames"""
        batch_images = []
        batch_scales = []
        batch_pads = []
        
        for frame in frames:
            image, scale, pads = self.preprocess_frame(frame)
            batch_images.append(image[0])  # Remove batch dim for stacking
            batch_scales.append(scale)
            batch_pads.append(pads)
        
        batch_tensor = np.stack(batch_images, axis=0)
        return batch_tensor, batch_scales, batch_pads
    
    def detect_single(self, frame, conf_thresh=0.5):
        """Detect objects in single frame"""
        if self.use_onnx:
            return self._detect_single_onnx(frame, conf_thresh)
        else:
            return self._detect_single_pytorch(frame, conf_thresh)
    
    def detect_batch(self, frames, conf_thresh=0.5):
        """Detect objects in batch of frames"""
        if self.use_onnx:
            return self._detect_batch_onnx(frames, conf_thresh)
        else:
            return self._detect_batch_pytorch(frames, conf_thresh)
    
    def _detect_single_pytorch(self, frame, conf_thresh):
        """PyTorch single frame detection"""
        results = self.model(frame, conf=conf_thresh, verbose=False)
        return self._format_pytorch_results(results[0])
    
    def _detect_batch_pytorch(self, frames, conf_thresh):
        """PyTorch batch detection"""
        results = self.model(frames, conf=conf_thresh, verbose=False)
        return [self._format_pytorch_results(result) for result in results]
    
    def _format_pytorch_results(self, result):
        """Format PyTorch YOLO results"""
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'confidence': float(confidences[i]),
                    'class_id': int(classes[i]),
                    'class_name': self.class_names[classes[i]]
                })
        return detections
    
    def _detect_single_onnx(self, frame, conf_thresh):
        """ONNX single frame detection"""
        preprocessed, scale, pads = self.preprocess_frame(frame)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: preprocessed})
        
        # Post-process
        return self._postprocess_onnx_output(outputs[0][0], conf_thresh, scale, pads, frame.shape)
    
    def _detect_batch_onnx(self, frames, conf_thresh):
        """ONNX batch detection"""
        batch_tensor, batch_scales, batch_pads = self.preprocess_batch(frames)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: batch_tensor})
        
        # Post-process each result
        batch_detections = []
        for i in range(len(frames)):
            detections = self._postprocess_onnx_output(
                outputs[0][i], conf_thresh, batch_scales[i], batch_pads[i], frames[i].shape
            )
            batch_detections.append(detections)
        
        return batch_detections
    
    def _postprocess_onnx_output(self, output, conf_thresh, scale, pads, orig_shape):
        """Post-process ONNX output to detections"""
        detections = []
        
        # YOLOv8 output format: [batch, 84, 8400] where 84 = 4(bbox) + 80(classes)
        output = output.transpose()  # [8400, 84]
        
        # Extract boxes and scores
        boxes = output[:, :4]
        scores = output[:, 4:]
        
        # Find best class for each detection
        class_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # Filter by confidence
        valid_indices = class_scores > conf_thresh
        
        if np.any(valid_indices):
            boxes = boxes[valid_indices]
            class_scores = class_scores[valid_indices]
            class_ids = class_ids[valid_indices]
            
            # Convert from center format to corner format
            x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Scale back to original image coordinates
            pad_x, pad_y = pads
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            
            # Clip to image bounds
            orig_height, orig_width = orig_shape[:2]
            x1 = np.clip(x1, 0, orig_width)
            y1 = np.clip(y1, 0, orig_height)
            x2 = np.clip(x2, 0, orig_width)
            y2 = np.clip(y2, 0, orig_height)
            
            # Format detections
            for i in range(len(x1)):
                detections.append({
                    'bbox': [float(x1[i]), float(y1[i]), float(x2[i]-x1[i]), float(y2[i]-y1[i])],
                    'confidence': float(class_scores[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': self.class_names[class_ids[i]]
                })
        
        return detections

def benchmark_detector(frames_dir, num_frames=100):
    """Benchmark different detector configurations"""
    frame_files = list(Path(frames_dir).glob('*.jpg'))[:num_frames]
    frames = [cv2.imread(str(f)) for f in frame_files]
    
    configs = [
        ('PyTorch CPU', {'use_gpu': False, 'use_onnx': False}),
        ('PyTorch GPU', {'use_gpu': True, 'use_onnx': False}),
        ('ONNX CPU', {'use_gpu': False, 'use_onnx': True}),
        ('ONNX GPU', {'use_gpu': True, 'use_onnx': True}),
    ]
    
    results = {}
    
    for name, config in configs:
        try:
            print(f"\nBenchmarking {name}...")
            detector = OptimizedDetector('yolov8s.pt', **config)
            
            # Warmup
            detector.detect_single(frames[0])
            
            # Benchmark
            start_time = time.time()
            total_detections = 0
            
            for frame in frames:
                detections = detector.detect_single(frame)
                total_detections += len(detections)
            
            elapsed = time.time() - start_time
            fps = len(frames) / elapsed
            
            results[name] = {
                'fps': fps,
                'total_detections': total_detections,
                'avg_detections_per_frame': total_detections / len(frames)
            }
            
            print(f"{name}: {fps:.1f} FPS, {total_detections} total detections")
            
        except Exception as e:
            print(f"{name}: Failed - {e}")
            results[name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    # Example usage
    detector = OptimizedDetector('yolov8s.pt', use_gpu=True, use_onnx=True)
    
    # Test on a sample frame
    frame = cv2.imread('data/keyframes/scene_001/cam_1/frame_000.jpg')
    if frame is not None:
        detections = detector.detect_single(frame)
        print(f"Detected {len(detections)} objects")
        for det in detections:
            print(f"  {det['class_name']}: {det['confidence']:.3f}")