import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_architecture():
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Define boxes [x, y, width, height]
    boxes = {
        'Raw Telemetry Data\\n(NetFlow, BGP, Optical)': [0.0, 0.4, 2.5, 0.5],
        'Sequence Generator\\n(15-step Rolling Window)': [3.0, 0.4, 2.5, 0.5],
        'LSTM Layer\\n(128 Hidden Units)': [6.0, 0.4, 2.0, 0.5],
        'Self-Attention Layer\\n(Dynamic Weighting)': [8.5, 0.4, 2.5, 0.5],
        'Dense Network\\n+ Sigmoid Activation': [11.5, 0.4, 2.5, 0.5],
        'Failure Prediction\\n(Soft/Hard Outage)': [14.5, 0.4, 2.5, 0.5]
    }
    
    for name, (x, y, w, h) in boxes.items():
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='#1f77b4', facecolor='#dcedf7', zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)
        
    # Draw arrows
    ordered_keys = list(boxes.keys())
    for i in range(len(ordered_keys) - 1):
        box1_x, box1_y, box1_w, box1_h = boxes[ordered_keys[i]]
        box2_x, box2_y, box2_w, box2_h = boxes[ordered_keys[i+1]]
        
        start_x = box1_x + box1_w
        start_y = box1_y + box1_h/2
        end_x = box2_x
        end_y = box2_y + box2_h/2
        
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", lw=2, color='#1f77b4'), zorder=1)

    plt.xlim(-0.5, 17.5)
    plt.ylim(0, 1.2)
    plt.axis('off')
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'architecture_diagram.png'), dpi=300, bbox_inches='tight')
    print("Generated Architecture Diagram.")

if __name__ == '__main__':
    draw_architecture()
