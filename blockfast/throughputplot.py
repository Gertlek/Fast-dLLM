import matplotlib.pyplot as plt
import numpy as np

# Performance data
data = {
    'Qwen2.5-7B-Instruct': {
        'throughput': 54.74,
        'accuracy': 82.0,
        'color': 'orange',
        'marker': 'o',
        'size': 150
    },
    'Qwen2.5-7B-Instruct + FlashAttnv2': {
        'throughput': 67.16,
        'accuracy': 82.0,
        'color': 'darkorange',
        'marker': 's',
        'size': 150
    },
    'Fast-dLLM v2 (7B)': {
        'throughput': 165.01,
        'accuracy': 79.0,
        'color': 'green',
        'marker': 'X',
        'size': 200
    },
    'Fast-dLLM v2 (7B) + FlashAttnv2': {
        'throughput': 182.84 ,
        'accuracy': 79.0,
        'color': 'darkgreen',
        'marker': '*',
        'size': 300
    },
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot each model
for model_name, model_data in data.items():
    ax.scatter(
        model_data['throughput'],
        model_data['accuracy'],
        c=model_data['color'],
        marker=model_data['marker'],
        s=model_data['size'],
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5,
        label=model_name
    )
    
    # Add model labels
    # Adjust offsets based on model position to avoid overlap
    if model_name == 'Qwen2.5-7B-Instruct':
        offset_x = 0
        offset_y = -12.5
    elif model_name == 'Qwen2.5-7B-Instruct + FlashAttnv2':
        offset_x = 0
        offset_y = 12.5
    elif model_name == 'Fast-dLLM v2 (7B)':
        offset_x = 0
        offset_y = 12.5
    elif model_name == 'Fast-dLLM v2 (7B) + FlashAttnv2':
        offset_x = 0
        offset_y = -12.5
    ax.annotate(
        model_name,
        (model_data['throughput'], model_data['accuracy']),
        xytext=(offset_x, offset_y),
        textcoords='offset points',
        fontsize=9,
        color=model_data['color'],
        fontweight='bold'
    )

# Add speedup annotation for Fast-dLLM v2
qwen_throughput = data['Qwen2.5-7B-Instruct']['throughput']
fastdllm_throughput = data['Fast-dLLM v2 (7B) + FlashAttnv2']['throughput']
speedup = fastdllm_throughput / qwen_throughput

# ax.annotate(
#     f'{speedup:.2f}× Faster',
#     xy=(fastdllm_throughput, data['Fast-dLLM v2 (7B) + FlashAttnv2']['accuracy']),
#     xytext=(60, 82.5),
#     fontsize=12,
#     color='green',
#     fontweight='bold',
#     arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='--')
# )

# # Add accuracy improvement annotation
# accuracy_improvement = data['Fast-dLLM v2 (7B) + FlashAttnv2']['accuracy'] - data['Qwen2.5-7B-Instruct']['accuracy']
# ax.annotate(
#     f'+{accuracy_improvement:.1f}%',
#     xy=(fastdllm_throughput, data['Fast-dLLM v2 (7B) + FlashAttnv2']['accuracy']),
#     xytext=(105, 81),
#     fontsize=12,
#     color='red',
#     fontweight='bold',
#     arrowprops=dict(arrowstyle='->', color='red', lw=2)
# )

# Set labels and title
ax.set_xlabel('Throughput (Tokens/sec)', fontsize=14, fontweight='bold')
ax.set_ylabel('GSM8K Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance: Throughput vs Accuracy', fontsize=16, fontweight='bold')

# Set axis limits
# ax.set_xlim(0, 2000)
ax.set_ylim(70, 86)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

# Tight layout
plt.tight_layout()

# Save and show
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'performance_comparison.png'")
plt.show()