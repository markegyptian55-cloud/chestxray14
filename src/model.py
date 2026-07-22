import os
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name="densenet121", num_classes=14, pretrained=True):
    """
    Factory function to get pre-trained models adjusted for multi-label classification.
    """
    print(f"Initializing {model_name} (pretrained={pretrained})...")
    
    # Map model names to their respective weight directories
    model_weight_dirs = {
        "densenet121": r"D:\project\DEEP LEARN PROJECT\NIH Chest X-rays\Dataset\pre-trained DenseNet121 small",
        "chexnet": r"D:\project\DEEP LEARN PROJECT\NIH Chest X-rays\Dataset\pre-trained CheXNet small",
        "resnet50": r"D:\project\DEEP LEARN PROJECT\NIH Chest X-rays\Dataset\pre-trained resnet50",
        "resnet18": r"D:\project\DEEP LEARN PROJECT\NIH Chest X-rays\Dataset\pre-trained DenseNet121 small", # fallback folder
        "densenet169": r"D:\project\DEEP LEARN PROJECT\NIH Chest X-rays\Dataset\pre-trained DenseNet121 small", # fallback folder
        "efficientnet_b4": r"D:\project\DEEP LEARN PROJECT\NIH Chest X-rays\Dataset\pre-trained efficientnet_b4 medium",
        "swin_t": r"D:\project\DEEP LEARN PROJECT\NIH Chest X-rays\Dataset\pre-trained Swin-T medium"
    }
    
    # Map model names to weight file names
    model_weight_files = {
        "densenet121": "densenet121-a639ec97.pth",
        "chexnet": "chexnet.pth.tar",
        "resnet50": "resnet50-0676ba61.pth",
        "resnet18": "resnet18-f37072fd.pth",
        "densenet169": "densenet169-b2777c96.pth",
        "efficientnet_b4": "efficientnet_b4_rwightman-23ab8bcd.pth",
        "swin_t": "swin_t-704ceda3.pth"
    }

    def load_base_model(name):
        # 1. Try to load local weights if they exist
        if pretrained and name in model_weight_files:
            local_dir = model_weight_dirs.get(name)
            local_path = os.path.join(local_dir, model_weight_files[name])
            if os.path.exists(local_path):
                print(f"Loading pre-trained weights locally from: {local_path}")
                
                # Instantiate base model architectures
                if name == "densenet121" or name == "chexnet":
                    model = models.densenet121(weights=None)
                elif name == "resnet50":
                    model = models.resnet50(weights=None)
                elif name == "resnet18":
                    model = models.resnet18(weights=None)
                elif name == "densenet169":
                    model = models.densenet169(weights=None)
                elif name == "efficientnet_b4":
                    model = models.efficientnet_b4(weights=None)
                elif name == "swin_t":
                    model = models.swin_t(weights=None)
                
                # Load state dict
                state_dict = torch.load(local_path, map_location="cpu", weights_only=False)
                
                # If loading CheXNet, unpack state_dict or strip nested keys and module prefix
                if name == "chexnet":
                    if isinstance(state_dict, dict) and "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_k = k
                        if new_k.startswith("module."):
                            new_k = new_k[7:]
                        if new_k.startswith("densenet121."):
                            new_k = new_k[12:]
                        new_state_dict[new_k] = v
                    state_dict = new_state_dict
                
                # Clean up DenseNet key naming issues between different torchvision versions
                if "densenet" in name or name == "chexnet":
                    import re
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_k = re.sub(r'(.denselayer\d+\.(?:norm|conv))\.(\d+)', r'\1\2', k)
                        new_state_dict[new_k] = v
                    state_dict = new_state_dict

                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if len(unexpected_keys) > 0:
                    # Filter classifier-related unexpected keys to print a short summary
                    clf_keys = [k for k in unexpected_keys if "classifier" in k]
                    if len(clf_keys) > 0:
                        print(f"Skipped loading pre-trained classifier keys (this is expected for fine-tuning): {clf_keys}")
                return model

        # 2. Fallback to normal download if local file is missing
        if name == "densenet121" or name == "chexnet":
            try:
                from torchvision.models import DenseNet121_Weights
                weights = DenseNet121_Weights.DEFAULT if pretrained else None
                return models.densenet121(weights=weights)
            except ImportError:
                return models.densenet121(pretrained=pretrained)
        elif name == "resnet50":
            try:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                return models.resnet50(weights=weights)
            except ImportError:
                return models.resnet50(pretrained=pretrained)
        elif name == "resnet18":
            try:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                return models.resnet18(weights=weights)
            except ImportError:
                return models.resnet18(pretrained=pretrained)
        elif name == "densenet169":
            try:
                from torchvision.models import DenseNet169_Weights
                weights = DenseNet169_Weights.DEFAULT if pretrained else None
                return models.densenet169(weights=weights)
            except ImportError:
                return models.densenet169(pretrained=pretrained)
        elif name == "efficientnet_b4":
            try:
                from torchvision.models import EfficientNet_B4_Weights
                weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
                return models.efficientnet_b4(weights=weights)
            except ImportError:
                return models.efficientnet_b4(pretrained=pretrained)
        elif name == "swin_t":
            try:
                from torchvision.models import Swin_T_Weights
                weights = Swin_T_Weights.DEFAULT if pretrained else None
                return models.swin_t(weights=weights)
            except ImportError:
                return models.swin_t(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown base model: {name}")

    model = load_base_model(model_name)

    # Modify classification layer for 14-disease multi-label output
    if model_name in ["densenet121", "densenet169", "chexnet"]:
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif model_name in ["resnet50", "resnet18"]:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "efficientnet_b4":
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == "swin_t":
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)
        
    return model

if __name__ == "__main__":
    # Quick sanity check
    model = get_model("densenet121", num_classes=14, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)  # Should be [2, 14]
