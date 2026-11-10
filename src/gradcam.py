import torch
import torch.nn.functional as F
import numpy as np

def make_gradcam(model, input_tensor, target_class=None):
    """Return a normalized 224x224 Grad-CAM heatmap for the last conv block."""
    activations = []
    gradients = []
    target_layer = model.features[-1][0]

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        score = output[:, target_class].sum()
        score.backward()

        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[0]).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam
    finally:
        h1.remove()
        h2.remove()
