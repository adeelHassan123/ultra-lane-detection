import torch

from models.baseline_cnn import BaselineCNN


def test_baseline_model_output_shape():
    model = BaselineCNN()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 1, 256, 256)
