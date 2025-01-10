# Understanding CNN Decisions with Grad-CAM

## Overview
This project implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize what Convolutional Neural Networks "see" when making predictions. As deep learning models often act as black boxes,
this implementation helps us understand their decision-making process by highlighting the (importance reions) regions of an image that most influence the model's predictions.

## What is Grad-CAM?
Grad-CAM is a visualization technique that provides insights into CNN's decision-making. It works by:
1. Taking the gradient of the predicted class score with respect to feature maps
2. Weighting these feature maps based on their importance
3. Combining them to create a heatmap showing which parts of the image influenced the prediction most

![Screenshot 2025-01-09 180046](https://github.com/user-attachments/assets/b630403f-fed7-48b2-8bac-0404623d4f3d)


## Project Journey & Learning Points

### Understanding the Core Concepts
- **Feature Maps**: Learned patterns in CNNs
- **Gradients**: How much each feature contributes to the final decision
- **Activation Maps**: Where in the image these features are found

### Key Implementation Challenges
1. **Hook Management**: 
   - Capturing gradients and features during forward/backward passes
   - Proper cleanup to prevent memory leaks

2. **Normalization**: 
   - Explored different normalization techniques
   - Understanding why we need both max and min-max normalization
   - Impact on visualization quality

3. **Visualization**: 
   - Converting activation maps to meaningful heatmaps
   - Proper overlay with original images
   - Color mapping for better interpretation

## Implementation

### Requirements
```bash
pip install torch torchvision pillow numpy opencv-python
```

### Basic Usage as included in run.ipynb
```python
from gradcam import GradCAM
import torchvision.models as models

# Load pre-trained VGG16
model = models.vgg16(pretrained=True)
target_layer = model.features[30]

# Initialize and use Grad-CAM
grad_cam = GradCAM(model, target_layer)
visualize_gradcam("path/to/image.jpg", model, target_layer)
```

### Understanding the Code

The implementation follows these key steps:

1. **Feature Extraction**:
```python
def _register_hooks(self):
    def save_features(module, input, output):
        self.features.append(output)
```
This captures intermediate feature maps during the forward pass.

2. **Gradient Computation**:
```python
class_score.backward()  # Compute gradients
weights = torch.mean(gradients, dim=(2, 3))  # Global average pooling
```
This shows how we compute the importance of each feature map.

3. **Visualization Generation**:
```python
cam = F.relu(cam)  # Apply ReLU to focus on features that have positive influence
```
We process the activation map to create interpretable visualizations.

## Results and Analysis

### Example 1 : 
The cell 3 in run.ipynb has output displayed as Afgan Hound and the outputs directory contains the result images,
Original Image:
![n02088094_97_original](https://github.com/user-attachments/assets/8484bb8d-3eac-4457-adc8-76bfd7339100)

The original gradient map obtained by processing gradients with their respective importance weights as per the channels,
![n02088094_97_grayscale](https://github.com/user-attachments/assets/5b3bdf7b-298b-4866-a854-e25dcde9927f)

The heatmap obtained by applying colormap using cv2
![n02088094_97_heatmap](https://github.com/user-attachments/assets/c5c9f20a-11a9-4684-a422-816577e9d504)

The overlay of heatmap on original image (640x640) shows that the model focuses on near face features as the class predicted is Afgan Hound, that is special kind of Hound Breed Dog.
![n02088094_97_overlay](https://github.com/user-attachments/assets/a7bf6c5d-b30b-4a04-a102-3c0a3365899b)


### Example 2: A basketball 🏀:
cell 8 in run.ipynb
The heatmap and overlay image shows that the model has focus in the stripes as it's not just a ball rather has a special kind.
Original Image:
![n02802426_12038_original](https://github.com/user-attachments/assets/a6a1ce72-d8a0-4ff8-99d0-637d3e398592)

The original gradient map obtained by processing gradients with their respective importance weights as per the channels:
![n03240683_15144_grayscale](https://github.com/user-attachments/assets/dc2013c0-2477-4209-ac9a-ecf9c5eb6c7a)

The processed heatmap obtained by applying colormap using cv2:
![n03240683_15144_heatmap](https://github.com/user-attachments/assets/e3a404f4-0de2-4c40-aae4-fb8ae3cbb85f)

The Overlay of heatmap over original image:
![n02802426_12038_overlay](https://github.com/user-attachments/assets/1f5e9d33-940c-437a-9ed6-eeb5a77c6443)




## Observations and Learning Insights

Through this Grad-CAM implementation, I gained practical insights into CNN behavior through hands-on experimentation:

### Layer Behavior Study
While implementing Grad-CAM, I was able to observe and confirm interesting patterns in model behavior:

1. **Model-Specific Patterns**: 
   - Even models trained for the same task can develop different internal representations
   - For example, in basketball images, I noticed the model's focus on the ball wasn't consistent throughout the layers
   - This hands-on observation helped me better understand how different architectures can arrive at similar results through different internal patterns

2. **Learning Through Visualization**:
   - Grad-CAM helped visualize how different layers process information
   - The practical implementation reinforced theoretical concepts about CNN behavior
   - Seeing these patterns firsthand made abstract concepts more concrete

This project served as an educational exercise in understanding CNN behavior through direct observation, helping bridge the gap between theoretical knowledge and practical implementation.
These are well known facts but the experimentation makes factual confirmations.

## Future Improvements
- [ ] Extend to other CNN architectures 
- [ ] Implement different visualization techniques
- [ ] Add support for batch processing


## Credits and Resources
- Original Grad-CAM paper: [Gradient-weighted Class Activation Mapping](https://arxiv.org/abs/1610.02391)
- PyTorch Documentation
- Images - [ImageNet Localization Challenge Image Dataset from akggle]


Feel free to reach out with questions or suggestions!
