from ultralytics import YOLO
import argparse
import random
# Load a visdrone-pretrained YOLOv8n model

parser = argparse.ArgumentParser(description='Vision Transformer in PyTorch')
parser.add_argument('--model', default='yolov8n.pt',
                    help='disables CUDA training')
parser.add_argument('--batch-size', type=int, default=16,
                    help='patch size for images (default : 16)')   

######### MINGEUN ###########
parser.add_argument('--dataset', default='VOC', help='used dataset')
parser.add_argument('--image-size', default='64', help='size of image for training if used')
parser.add_argument('--workers', type=int, default=16)
######### MINGEUN ###########
              
args = parser.parse_args()

model = YOLO(args.model)
args.image_size = 320

job_name = (args.model).strip(".pt")

# Display model information (optional)

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='{}.yaml'.format(args.dataset), epochs=100, imgsz=args.image_size, batch=args.batch_size, project=job_name, workers=args.workers)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
# results = model('path/to/bus.jpg')