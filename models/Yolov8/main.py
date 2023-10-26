from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model

parser = argparse.ArgumentParser(description='Vision Transformer in PyTorch')
parser.add_argument('--model', default='yolov8n.pt',
                    help='disables CUDA training')
parser.add_argument('--batch-size', type=int, default=16,
                    help='patch size for images (default : 16)')
args = parser.parse_args()

model = YOLO(args.model)

# Display model information (optional)

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='VisDrone.yaml', epochs=100, imgsz=640, batch=args.batch_size)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
# results = model('path/to/bus.jpg')