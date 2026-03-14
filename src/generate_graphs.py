import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def generate_f1_comparison():
    # Model Names
    models = ['Logistic Regression', 'Random Forest', 'Attention-LSTM (Optical)', 'Attention-LSTM (Cisco)', 'Attention-LSTM (Omni)']
    
    # 60s Window F1-Scores sourced from our experimental validation
    f1_scores = [0.1791, 0.2275, 0.3737, 0.4318, 0.3791]
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create the bar chart
    ax = sns.barplot(x=models, y=f1_scores, palette="viridis")
    
    # Add labels and titles
    plt.title('Baseline vs Attention-LSTM Performance (60-sec Window F1-Score)', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12)
    plt.xlabel('Model Architecture & Validation Set', fontsize=12)
    plt.ylim(0, 0.5)
    
    # Add numerical labels on top of bars
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.01, str(round(v, 4)), ha='center', fontweight='bold', fontsize=11)
        
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'f1_comparison_chart.png'), dpi=300)
    print("Generated F1 Comparison Chart.")

def generate_loss_curves():
    # Epoch Loss values from our final console output logs
    optical_loss = [0.2698, 0.1546, 0.1171, 0.1096, 0.1071, 0.1075, 0.1047, 0.0970, 0.0953, 0.0953, 0.0946, 0.0808]
    cisco_loss = [0.4996, 0.3457, 0.3245, 0.3071, 0.2973, 0.2886]
    omni_loss = [0.5546, 0.3405, 0.3130, 0.2956, 0.2846, 0.2769]
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot curves
    plt.plot(range(1, 13), optical_loss, marker='o', linewidth=2, label='Optical (Soft Failure)')
    plt.plot(range(1, 7), cisco_loss, marker='s', linewidth=2, label='Cisco (BGP Anomalies)')
    plt.plot(range(1, 7), omni_loss, marker='^', linewidth=2, label='Omni-Architecture (Combined)')
    
    plt.title('Attention-LSTM Training Loss Convergence (BCEWithLogitsLoss)', fontsize=14, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=12)
    plt.xlabel('Epochs', fontsize=12)
    plt.xticks(range(1, 13))
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    plt.savefig(os.path.join(output_dir, 'training_loss_convergence.png'), dpi=300)
    print("Generated Training Loss Curve.")

if __name__ == '__main__':
    generate_f1_comparison()
    generate_loss_curves()
