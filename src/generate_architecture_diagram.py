import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_architecture():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define boxes [x, y, width, height] in a grid layout to maximize space
    # Two rows of three boxes
    w, h = 3.0, 1.0 # Width and height of boxes
    spacing_x = 1.0
    spacing_y = 1.5
    
    # Top row
    x1 = 0
    y1 = spacing_y * 1.5
    
    # Bottom row
    x2 = 0
    y2 = 0
    
    boxes = {
        '1. Raw Telemetry\\nData\\n(NetFlow, Optical)': [0, y1, w, h],
        '2. Sequence\\nGenerator\\n(15-step Window)': [w + spacing_x, y1, w, h],
        '3. LSTM Layer\\n(128 Hidden Units)': [2*(w + spacing_x), y1, w, h],
        '4. Self-Attention\\nMechanism\\n(Dynamic Weighting)': [2*(w + spacing_x), y2, w, h],
        '5. Dense Network\\n+ Sigmoid\\nActivation': [w + spacing_x, y2, w, h],
        '6. Early-Warning\\nFailure Prediction\\n(Soft/Hard Outage)': [0, y2, w, h]
    }
    
    # Box styling
    edge_color = '#2c3e50'
    face_color = '#ecf0f1'
    highlight_face = '#d6eaf8'
    
    for i, (name, (x, y, box_w, box_h)) in enumerate(boxes.items()):
        # Highlight the Attention and LSTM layers
        current_face = highlight_face if 'LSTM' in name or 'Attention' in name else face_color
        
        # Draw box
        rect = patches.Rectangle((x, y), box_w, box_h, linewidth=2, edgecolor=edge_color, facecolor=current_face, zorder=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x + box_w/2, y + box_h/2, name, ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50', zorder=3)
        
    # Draw arrows
    ordered_keys = list(boxes.keys())
    arrow_color = '#e74c3c'
    
    for i in range(len(ordered_keys) - 1):
        box1_name = ordered_keys[i]
        box2_name = ordered_keys[i+1]
        
        x1, y1, w1, h1 = boxes[box1_name]
        x2, y2, w2, h2 = boxes[box2_name]
        
        if y1 == y2: 
            # Horizontal parsing
            if x1 < x2:
                # Left to right
                start_x = x1 + w1
                start_y = y1 + h1/2
                end_x = x2
                end_y = y2 + h2/2
            else:
                # Right to left (bottom row)
                start_x = x1
                start_y = y1 + h1/2
                end_x = x2 + w2
                end_y = y2 + h2/2
                
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle="->", lw=3, color=arrow_color), zorder=1)
        else:
            # Vertical transition (From LSTM down to Attention)
            start_x = x1 + w1/2
            start_y = y1
            end_x = x2 + w2/2
            end_y = y2 + h2
            
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle="->", lw=3, color=arrow_color), zorder=1)

    # Set boundaries
    plt.xlim(-0.5, 3*(w + spacing_x) - spacing_x + 0.5)
    plt.ylim(-0.5, y1 + h + 0.5)
    plt.axis('off')
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'architecture_diagram.png'), dpi=300, bbox_inches='tight')
    print("Generated Improved Architecture Diagram.")

if __name__ == '__main__':
    draw_architecture()
