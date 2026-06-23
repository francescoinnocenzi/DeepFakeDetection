import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data extracted from notebook training log
epochs = list(range(1, 10))
train_losses = [0.3124, 0.1454, 0.0994, 0.0794, 0.0646, 0.0563, 0.0512, 0.0443, 0.0437]
val_losses = [0.1826, 0.1367, 0.1290, 0.1125, 0.1354, 0.1107, 0.1175, 0.1162, 0.1155]

plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='#ff6f61', linewidth=2.5)
plt.plot(epochs, val_losses, label='Validation Loss', marker='s', color='#5ebbe4', linewidth=2.5)

# Styling
plt.title('Multi-Task Loss Curve (Alpha=0.5, Beta=0.5)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Joint Loss Value', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)

# Annotate early stopping trigger point
plt.annotate('Early Stopping Triggered', xy=(9, 0.1155), xytext=(6, 0.15),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))

# Save the plot
plt.savefig('overfitting_curve.png', bbox_inches='tight', dpi=300)
print("Overfitting curve plot saved successfully as 'overfitting_curve.png'.")
plt.show()
