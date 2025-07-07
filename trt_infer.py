import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
# import pycuda.autoinit # REMOVED: Manually manage CUDA context
import colorsys

class TensorRTInference:
    def __init__(self, engine_path):
        # Initialize CUDA context
        # Ensure the device is available and create context
        if cuda.Device.count() == 0:
            raise RuntimeError("No CUDA devices found. Cannot initialize TensorRT.")
        self.cuda_ctx = cuda.Device(0).make_context()
        print("TensorRTInference: CUDA context created.")
        
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")
        
        self.context = self.engine.create_execution_context()
        print("TensorRTInference: Engine and execution context created.")
        
        # Allocate GPU memory for inputs and outputs
        self.allocate_buffers()
        
        # Get input/output shapes
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)
        
        # Get input dimensions for preprocessing
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        
        # Determine YOLO version and output format
        self.determine_yolo_format()
        
        # Generate colors for each class
        self.colors = self._generate_colors(self.num_classes)
        
        # Default class names (modify for your custom classes)
        self.class_names = self._get_default_class_names()
        print("TensorRTInference: Initialization complete.")
    
    def allocate_buffers(self):
        """Allocate GPU memory for inputs and outputs"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.engine.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to device bindings
            self.bindings.append(int(device_mem))
            
            # Append to the appropriate list
            if self.engine.binding_is_input(binding_idx):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def determine_yolo_format(self):
        """Determine YOLO format based on output shape"""
        if len(self.output_shape) == 3:
            feature_dim = self.output_shape[1]
            if feature_dim < 20:
                self.yolo_version = 'v8'
                self.num_classes = feature_dim - 4
                self.num_anchors = self.output_shape[2]
            else:
                self.yolo_version = 'v5'
                self.num_classes = self.output_shape[2] - 5
                self.num_anchors = self.output_shape[1]
        else:
            self.yolo_version = 'v5'
            self.num_classes = 4
    
    def _generate_colors(self, num_classes):
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append([int(c * 255) for c in rgb])
        return colors
    
    def _get_default_class_names(self):
        custom_classes = {
            0: "inclusion",
            1: "patches",
            2: "pitted_surface", 
            3: "scratches"
        }
        if self.num_classes == 4:
            return custom_classes
        else:
            class_names = {}
            for i in range(self.num_classes):
                if i in custom_classes:
                    class_names[i] = custom_classes[i]
                else:
                    class_names[i] = f"class_{i}"
            return class_names
    
    def preprocess(self, image):
        self.orig_h, self.orig_w = image.shape[:2]
        resized = cv2.resize(image, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        return batched
    
    def postprocess_v8(self, output, conf_threshold=0.5, nms_threshold=0.4):
        output = output[0].T
        boxes, scores, class_ids = [], [], []
        max_coord = np.max(output[:, :4])
        is_normalized = max_coord <= 1.0
        x_scale = self.orig_w if is_normalized else self.orig_w / self.input_w
        y_scale = self.orig_h if is_normalized else self.orig_h / self.input_h
        for detection in output:
            x_center, y_center, width, height = detection[:4]
            class_scores = detection[4:4+self.num_classes]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            if confidence > conf_threshold:
                if is_normalized:
                    x_center *= self.orig_w
                    y_center *= self.orig_h
                    width *= self.orig_w
                    height *= self.orig_h
                else:
                    x_center *= x_scale
                    y_center *= y_scale
                    width *= x_scale
                    height *= y_scale
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                x1 = max(0, min(x1, self.orig_w))
                y1 = max(0, min(y1, self.orig_h))
                x2 = max(0, min(x2, self.orig_w))
                y2 = max(0, min(y2, self.orig_h))
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(int(class_id))
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                return [boxes[i] for i in indices], [scores[i] for i in indices], [class_ids[i] for i in indices]
        return [], [], []
    
    def postprocess_v5(self, output, conf_threshold=0.5, nms_threshold=0.4):
        if len(output.shape) == 3:
            output = output[0]
        if output.shape[0] > output.shape[1]:
            output = output.T
        boxes, scores, class_ids = [], [], []
        max_coord = np.max(output[:4, :])
        is_normalized = max_coord <= 1.0
        x_scale = self.orig_w if is_normalized else self.orig_w / self.input_w
        y_scale = self.orig_h if is_normalized else self.orig_h / self.input_h
        for detection in output.T:
            objectness = detection[4]
            if objectness > conf_threshold:
                class_scores = detection[5:5+self.num_classes] * objectness
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                if confidence > conf_threshold:
                    center_x = detection[0]
                    center_y = detection[1]
                    width = detection[2]
                    height = detection[3]
                    if is_normalized:
                        center_x *= self.orig_w
                        center_y *= self.orig_h
                        width *= self.orig_w
                        height *= self.orig_h
                    else:
                        center_x *= x_scale
                        center_y *= y_scale
                        width *= x_scale
                        height *= y_scale
                    x1 = int(center_x - width / 2)
                    y1 = int(center_y - height / 2)
                    x2 = int(center_x + width / 2)
                    y2 = int(center_y + height / 2)
                    x1 = max(0, min(x1, self.orig_w))
                    y1 = max(0, min(y1, self.orig_h))
                    x2 = max(0, min(x2, self.orig_w))
                    y2 = max(0, min(y2, self.orig_h))
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence))
                    class_ids.append(int(class_id))
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                return [boxes[i] for i in indices], [scores[i] for i in indices], [class_ids[i] for i in indices]
        return [], [], []
    
    def postprocess(self, output, conf_threshold=0.5, nms_threshold=0.4):
        if self.yolo_version == 'v8':
            return self.postprocess_v8(output, conf_threshold, nms_threshold)
        else:
            return self.postprocess_v5(output, conf_threshold, nms_threshold)
    
    def draw_detections(self, image, boxes, scores, class_ids):
        result_image = image.copy()
        label_positions = []  # Keep track of label y-ranges and x-ranges

        def is_light(color):
            # Simple luminance check (OpenCV uses BGR)
            return (0.299*color[2] + 0.587*color[1] + 0.114*color[0]) > 186

        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.orig_w, x2)
            y2 = min(self.orig_h, y2)
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Default label position above the box
            label_y1 = y1 - text_height - baseline - 5
            label_y2 = y1
            draw_x1 = x1
            if label_y1 < 0:
                label_y1 = y1
                label_y2 = y1 + text_height + baseline + 5
                text_org = (draw_x1, label_y2 - baseline - 2)
            else:
                text_org = (draw_x1, y1 - baseline - 2)
            label_x2 = draw_x1 + text_width
            if label_x2 > result_image.shape[1] - 1:
                shift = label_x2 - (result_image.shape[1] - 1)
                draw_x1 = x1 - shift
                if draw_x1 < 0:
                    draw_x1 = 0
                label_x2 = draw_x1 + text_width
                text_org = (draw_x1, text_org[1])

            # --- Prevent overlap with previous labels ---
            margin = 2
            while any(
                abs(label_y1 - prev_y2) < (text_height + baseline + margin)
                and not (label_x2 < prev_x1 or draw_x1 > prev_x2)
                for (prev_y1, prev_y2, prev_x1, prev_x2) in label_positions
            ):
                # Shift label down by its height + margin
                label_y1 += text_height + baseline + margin
                label_y2 += text_height + baseline + margin
                text_org = (draw_x1, label_y2 - baseline - 2 if label_y1 > y1 else y1 - baseline - 2)
                if label_y2 > result_image.shape[0]:
                    break

            label_positions.append((label_y1, label_y2, draw_x1, label_x2))
            # Choose text color based on background
            text_color = (0, 0, 0) if is_light(color) else (255, 255, 255)
            cv2.rectangle(
                result_image,
                (draw_x1, label_y1),
                (label_x2, label_y2),
                color,
                -1
            )
            cv2.putText(
                result_image,
                label,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2
            )
        return result_image
    
    def infer(self, image):
        input_tensor = self.preprocess(image)
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        output = self.outputs[0]['host'].reshape(self.output_shape)
        return output
    
    def infer_and_visualize(self, image, conf_threshold=0.5, nms_threshold=0.4, save_path=None):
        border = 40  # Add 40px border to all sides
        image_with_border = cv2.copyMakeBorder(
            image, border, border, border, border,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        output = self.infer(image)
        boxes, scores, class_ids = self.postprocess(output, conf_threshold, nms_threshold)
        # Offset all boxes by the border size
        boxes_offset = [[x1+border, y1+border, x2+border, y2+border] for (x1, y1, x2, y2) in boxes]
        result_image = self.draw_detections(image_with_border, boxes_offset, scores, class_ids)
        if save_path:
            cv2.imwrite(save_path, result_image)
        return result_image, boxes, scores, class_ids
    
    def __del__(self):
        self.destroy()

    def destroy(self):
        """Explicitly destroy the TensorRT engine and CUDA context."""
        print("TensorRTInference: Destroying resources...")
        try:
            if hasattr(self, 'context') and self.context:
                del self.context
                self.context = None
                print("TensorRTInference: Execution context destroyed.")
            if hasattr(self, 'engine') and self.engine:
                del self.engine
                self.engine = None
                print("TensorRTInference: Engine destroyed.")
            if hasattr(self, 'stream') and self.stream:
                del self.stream
                self.stream = None
                print("TensorRTInference: CUDA stream destroyed.")

            # Free allocated buffers
            for inp in getattr(self, 'inputs', []):
                if 'device' in inp and inp['device']:
                    cuda.mem_free(inp['device'])
                if 'host' in inp and inp['host'] is not None:
                    del inp['host']
            for out in getattr(self, 'outputs', []):
                if 'device' in out and out['device']:
                    cuda.mem_free(out['device'])
                if 'host' in out and out['host'] is not None:
                    del out['host']
            self.inputs = []
            self.outputs = []
            self.bindings = []
            print("TensorRTInference: Buffers freed.")
        except Exception as e:
            print(f"TensorRTInference: Error during resource destruction: {e}")
        finally:
            # Pop the CUDA context last, even if there was an error above
            try:
                if hasattr(self, 'cuda_ctx') and self.cuda_ctx:
                    current_ctx = cuda.Context.get_current()
                    if current_ctx == self.cuda_ctx:
                        self.cuda_ctx.pop()
                        print("TensorRTInference: CUDA context popped.")
                    else:
                        print("TensorRTInference: CUDA context not current, skipping pop.")
                    del self.cuda_ctx
                    self.cuda_ctx = None
            except Exception as e:
                print(f"TensorRTInference: Error popping CUDA context in finally: {e}")
        print("TensorRTInference: Resources destruction complete.")