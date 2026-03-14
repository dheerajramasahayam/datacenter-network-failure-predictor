import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_architecture():
    # A purely vertical layout fits perfectly in a single IEEE column (approx 3.5 inches wide)
    fig, ax = plt.subplots(figsize=(4, 8))
    
    # 6 boxes stacked vertically, reading top to bottom
    w, h = 3.6, 0.8 # Width and height of boxes
    spacing_y = 1.3
    
    x = 0.2
    
    boxes = [
        ('1. Raw Telemetry Data\\n(NetFlow, BGP, Optical)', 5 * spacing_y),
        ('2. Sequence Generator\\n(15-step Rolling Window)', 4 * spacing_y),
        ('3. LSTM Layer\\n(128 Hidden Units)', 3 * spacing_y),
        ('4. Self-Attention Mechanism\\n(Dynamic Weighting)', 2 * spacing_y),
        ('5. Dense Network\\n+ Sigmoid Activation', 1 * spacing_y),
        ('6. Early-Warning Failure Prediction\\n(Soft/Hard Outage)', 0)
    ]
    
    # Box styling
    edge_color = '#2c3e50'
    face_color = '#ecf0f1'
    highlight_face = '#d6eaf8'
    
    for i, (name, y) in enumerate(boxes):
        current_face = highlight_face if 'LSTM' in name or 'Attention' in name else face_color
        
        # Draw box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=edge_color, facecolor=current_face, zorder=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', fontsize=10, fontweight='bold', color='#2c3e50', zorder=3)
        
    # Draw arrows going straight down
    arrow_color = '#e74c3c'
    
    for i in range(len(boxes) - 1):
        _, y1 = boxes[i]
        _, y2 = boxes[i+1]
        
        start_x = x + w/2
        start_y = y1
        end_x = x + w/2
        end_y = y2 + h
        
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", lw=3, color=arrow_color), zorder=1)

    # Set boundaries
    plt.xlim(0, 4)
    plt.ylim(-0.2, 5 * spacing_y + h + 0.2)
    plt.axis('off')
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'architecture_diagram.png'), dpi=300, bbox_inches='tight')
    print("Generated Vertical Architecture Diagram.")

if __name__ == '__main__':
    draw_architecture()
