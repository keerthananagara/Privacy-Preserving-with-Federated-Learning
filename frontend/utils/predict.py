import torch
from torchvision import transforms
from PIL import Image
from MLModel import ScatterLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_digit(image_path, model_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    model = ScatterLinear().to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Fix state_dict keys
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("global_model."):
            new_state_dict[k.replace("global_model.", "")] = v
        else:
            new_state_dict[k] = v

    # Load fixed state_dict
    model.load_state_dict(new_state_dict)
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
