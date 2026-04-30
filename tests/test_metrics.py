from metrics import iou_score, dice_score

# Test with perfect prediction
pred = torch.ones(2, 1, 256, 256).cuda()
target = torch.ones(2, 1, 256, 256).cuda()

iou = iou_score(pred, target)
dice = dice_score(pred, target)

print(f"IoU (perfect): {iou:.4f}")   # Should be ~1.0
print(f"Dice (perfect): {dice:.4f}") # Should be ~1.0

# Test with all-zero prediction
pred = torch.zeros(2, 1, 256, 256).cuda()
target = torch.ones(2, 1, 256, 256).cuda()

iou = iou_score(pred, target)
print(f"IoU (all wrong): {iou:.4f}")  # Should be ~0.0

print("✅ Metrics working")