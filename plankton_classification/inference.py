import os
import json
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from GR_ViT import GRViT_mini as create_model  
from utils import evaluate  




def load_model(weights_path, num_classes, device):
    model = create_model(num_classes=num_classes).to(device)
    assert os.path.exists(weights_path), "weights file: '{}' not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    img_size = 224  
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    data_root = os.path.abspath(os.path.join(os.getcwd(), "."))  # get data root path
    image_path = os.path.join(data_root, "WHOI_custom")  # dataset path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform)

    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            pin_memory=True, 
                            num_workers=args.num_workers)

    # Load the model
    model = load_model(args.weights, args.num_classes, device)

    # Evaluate the model
    val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=0)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=79)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weights', type=str, default='./weight.pth')  # give the weight file path
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_workers', type=int, default=4)

    opt = parser.parse_args()

    main(opt)