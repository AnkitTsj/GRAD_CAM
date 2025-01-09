import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms
import urllib.request
import json
import os
import numpy as np
import cv2
from typing import Tuple, List, Optional


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: List[torch.Tensor] = []
        self.features: List[torch.Tensor] = []
        self._register_hooks()

    def _register_hooks(self):
        def save_gradients(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        def save_features(module, input, output):
            self.features.append(output)

        self.forward_hook = self.target_layer.register_forward_hook(save_features)
        self.backward_hook = self.target_layer.register_backward_hook(save_gradients)

    def _clean_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
        self.gradients.clear()
        self.features.clear()

    def load_and_preprocess_image(self, image_path: str, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """Load and preprocess an image."""
        try:
            image = Image.open(image_path)
            transformer = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            return transformer(image).unsqueeze(0)
        except Exception as e:
            raise RuntimeError(f"Error processing image: {str(e)}")

    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """Generate the Class Activation Map."""
        self.model.eval()
        with torch.set_grad_enabled(True):
            # Forward pass
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
                print("Target_class : ", target_class)

            # Get predicted class name
            class_name = load_imagenet_labels()[target_class]
            print(f"Predicted class: {class_name}")

            # Backward pass
            self.model.zero_grad()
            class_score = output[0][target_class]
            class_score.backward()

            # Generate CAM
            gradients = self.gradients[0]
            features = self.features[0]
            weights = torch.mean(gradients, dim=(2, 3))
            cam = torch.zeros(features.shape[2:], dtype=torch.float32)

            for i, w in enumerate(weights[0]):
                cam += w * features[0, i, :, :]

            cam = F.relu(cam)

            # Normalize CAM
            if cam.max() > 0:
                cam = cam / cam.max()

            self._clean_hooks()
            return cam


def apply_heatmap(cam: torch.Tensor, original_img: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Apply heatmap overlay on original image."""
    # Normalize CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cam.detach().numpy()

    # Create heatmap
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    # print(original_img.shape)
    # Convert to RGB if necessary
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    elif original_img.shape[2] == 4:  # Handle RGBA images
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)

    # Ensure same size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Create overlay
    overlay = cv2.addWeighted(
        original_img,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay, heatmap


def load_imagenet_labels() -> List[str]:
    """Load ImageNet class labels."""
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        response = urllib.request.urlopen(url)
        return json.loads(response.read())
    except Exception as e:
        print(f"Error loading ImageNet labels: {e}")
        return [""] * 1000  # Return empty labels if loading fails


def visualize_gradcam(image_path: str, model: nn.Module, target_layer: nn.Module,
                      output_size: Tuple[int, int] = (640, 640),
                      output_dir: Optional[str] = None,if_show = False) -> None:
    """
    Generate and save Grad-CAM visualizations.

    Args:
        image_path: Path to input image
        model: PyTorch model
        target_layer: Layer to generate CAM for
        output_size: Size of output images
        output_dir: Directory to save outputs (optional)
    """
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)

    # Load and preprocess image
    input_tensor = grad_cam.load_and_preprocess_image(image_path)
    original_img = Image.open(image_path).resize(output_size)
    original_img_array = np.array(original_img)

    # Generate CAM
    cam = grad_cam.generate_cam(input_tensor)

    # Resize CAM to match original image
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=output_size,
        mode='bilinear'
    ).squeeze()

    # Generate visualizations
    overlay, heatmap = apply_heatmap(cam, original_img_array)

    # Convert to PIL images
    overlay_img = Image.fromarray(overlay)
    heatmap_img = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    grayscale_cam = Image.fromarray((cam.detach().numpy() * 255).astype(np.uint8))

    # Save images if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        original_img.save(os.path.join(output_dir, f"{base_name}_original.png"))
        overlay_img.save(os.path.join(output_dir, f"{base_name}_overlay.png"))
        heatmap_img.save(os.path.join(output_dir, f"{base_name}_heatmap.png"))
        grayscale_cam.save(os.path.join(output_dir, f"{base_name}_grayscale.png"))

    # Display images
    if if_show:
        original_img.show(title="Original Image")
        overlay_img.show(title="Grad-CAM Overlay")
        heatmap_img.show(title="Heatmap")
        grayscale_cam.show(title="Grayscale CAM")

