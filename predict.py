import os
import yaml
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.fcn_unet import FCNUnet

def load_model(config, checkpoint_path):
    model = FCNUnet(
        num_classes=config['model']['num_classes'],
        in_channels=config['model']['in_channels'],
        feature_channels=config['model']['feature_channels']
    )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def preprocess_image(image_path, img_size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    return image, image_tensor.unsqueeze(0)

def postprocess_output(output, color_map):
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    height, width = pred.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        color_mask[pred == class_id] = color
    
    return color_mask

def overlay_mask(image, mask, alpha=0.5):
    image_np = np.array(image)
    mask = Image.fromarray(mask)
    mask = mask.resize(image_np.shape[:2])
    mask_np = np.array(mask)
    
    overlay = image_np.copy()
    overlay[mask_np != 0] = overlay[mask_np != 0] * (1 - alpha) + mask_np[mask_np != 0] * alpha
    overlay = overlay.astype(np.uint8)
    
    return overlay

def main():
    # Load config
    with open('configs/voc_seg.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create color map (class ID to RGB)
    color_map = {
        0: [0, 0, 0],          # background
        1: [128, 0, 0],        # aeroplane
        2: [0, 128, 0],        # bicycle
        3: [128, 128, 0],      # bird
        4: [0, 0, 128],        # boat
        5: [128, 0, 128],      # bottle
        6: [0, 128, 128],      # bus
        7: [128, 128, 128],    # car
        8: [64, 0, 0],         # cat
        9: [192, 0, 0],        # chair
        10: [64, 128, 0],      # cow
        11: [192, 128, 0],     # diningtable
        12: [64, 0, 128],      # dog
        13: [192, 0, 128],     # horse
        14: [64, 128, 128],    # motorbike
        15: [192, 128, 128],   # person
        16: [0, 64, 0],        # pottedplant
        17: [128, 64, 0],      # sheep
        18: [0, 192, 0],       # sofa
        19: [128, 192, 0],     # train
        20: [0, 64, 128]       # tvmonitor
    }
    
    # Load model
    model = load_model(config, 'checkpoints/best_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Test image path
    test_image_path = 'data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'
    
    # Preprocess
    image, image_tensor = preprocess_image(test_image_path, config['data']['img_size'])
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
    
    # Postprocess
    color_mask = postprocess_output(output, color_map)
    overlay = overlay_mask(image, color_mask)
    
    # Save results
    os.makedirs('output', exist_ok=True)
    Image.fromarray(color_mask).save('output/segmentation_mask.png')
    Image.fromarray(overlay).save('output/overlay.png')
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(color_mask)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('output/result.png')
    print('Prediction completed. Results saved in output/ directory.')

if __name__ == '__main__':
    main()