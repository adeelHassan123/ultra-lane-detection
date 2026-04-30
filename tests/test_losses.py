# In Colab, test that losses work
import torch
from losses import get_loss

# Create dummy tensors
logits = torch.randn(2, 1, 256, 256).cuda()
masks = torch.randint(0, 2, (2, 1, 256, 256)).float().cuda()

# Test Tversky Loss
tversky = get_loss('tversky', alpha=0.3, beta=0.7)
loss = tversky(logits, masks)
print(f"Tversky Loss: {loss.item():.4f}")  # Should show a number, not error

# Test Combined Loss
combined = get_loss('combined', pos_weight=10.0)
loss = combined(logits, masks)
print(f"Combined Loss: {loss.item():.4f}")

print("✅ Loss functions working")