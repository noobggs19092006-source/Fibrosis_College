import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Formatting
plt.style.use('grayscale')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

out_dir = "publication_figures"
os.makedirs(out_dir, exist_ok=True)

def draw_box(ax, x, y, width, height, text, fontsize=12):
    box = patches.Rectangle((x - width/2, y - height/2), width, height,
                            fill=True, color='white', ec='black', lw=2, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color='black'), zorder=1)

# ==========================================
# Figure 1: Overall Workflow
# ==========================================
def fig1_workflow():
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    w, h = 4, 1
    cx = 5
    
    y_pos = [13, 11, 9, 7, 5]
    texts = ["ChEMBL Dataset", "Data Cleaning", "Murcko Scaffold Split", 
             "Feature Preprocessing", "Feature Selection"]
    
    for i in range(len(y_pos)):
        draw_box(ax, cx, y_pos[i], w, h, texts[i])
        if i < len(y_pos) - 1:
            draw_arrow(ax, cx, y_pos[i] - h/2, cx, y_pos[i+1] + h/2)
            
    # Models branch
    draw_arrow(ax, cx, y_pos[-1] - h/2, cx, 3 + h/2)
    draw_arrow(ax, cx, y_pos[-1] - h/2 - 0.5, 2, 3 + h/2)
    draw_arrow(ax, cx, y_pos[-1] - h/2 - 0.5, 8, 3 + h/2)
    
    ax.plot([2, 8], [y_pos[-1] - h/2 - 0.5, y_pos[-1] - h/2 - 0.5], color='black', lw=2)

    draw_box(ax, 2, 3, 2.5, h, "Random Forest")
    draw_box(ax, 5, 3, 2.5, h, "XGBoost")
    draw_box(ax, 8, 3, 2.5, h, "ASNN")
    
    # Merge
    ax.plot([2, 8], [2.5, 2.5], color='black', lw=2)
    draw_arrow(ax, 2, 3 - h/2, 2, 2.5)
    draw_arrow(ax, 8, 3 - h/2, 8, 2.5)
    draw_arrow(ax, 5, 3 - h/2, 5, 1.5)
    
    draw_box(ax, cx, 1, 4, 0.8, "Consensus Prediction")
    draw_arrow(ax, cx, 0.6, cx, -0.4)
    draw_box(ax, cx, -0.8, 5, 0.8, "Classification + Regression")
    draw_arrow(ax, cx, -1.2, cx, -2.2)
    draw_box(ax, cx, -2.6, 5, 0.8, "SHAP + AD + Validation")
    
    ax.set_ylim(-4, 14)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_1_Overall_Workflow.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# Figure 2: Scaffold-based Dataset Split
# ==========================================
def fig2_scaffold_split():
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    cx = 5
    w, h = 5, 1.2
    
    draw_box(ax, cx, 9, w, h, "Original Dataset")
    draw_arrow(ax, cx, 9 - h/2, cx, 6.5 + h/2)
    draw_box(ax, cx, 6.5, w, h, "Murcko Scaffold Extraction")
    
    draw_arrow(ax, cx, 6.5 - h/2, cx, 4.5)
    ax.plot([2, 8], [4.5, 4.5], color='black', lw=2)
    
    draw_arrow(ax, 2, 4.5, 2, 3 + h/2)
    draw_arrow(ax, 5, 4.5, 5, 3 + h/2)
    draw_arrow(ax, 8, 4.5, 8, 3 + h/2)
    
    draw_box(ax, 2, 3, 2.5, h, "70% Train\n(No scaffold overlap)", fontsize=10)
    draw_box(ax, 5, 3, 2.5, h, "15% Validation\n(No scaffold overlap)", fontsize=10)
    draw_box(ax, 8, 3, 2.5, h, "15% Test\n(No scaffold overlap)", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_2_Scaffold_Split.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# Figure 3: Feature Preprocessing Pipeline
# ==========================================
def fig3_preprocessing():
    fig, ax = plt.subplots(figsize=(5, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    cx = 5
    w, h = 6, 1
    y_pos = np.linspace(13, 1, 7) if 'np' in globals() else [13, 11, 9, 7, 5, 3, 1]
    
    texts = ["343 descriptors", "Median Imputation", "Variance Filter", 
             "Mutual Information", "Correlation Removal", "Standardization", "118 descriptors"]
    
    for i in range(len(y_pos)):
        if i == 0 or i == 6:
            draw_box(ax, cx, y_pos[i], w, h, texts[i], fontsize=14)
        else:
            draw_box(ax, cx, y_pos[i], w, h, texts[i])
            
        if i < len(y_pos) - 1:
            draw_arrow(ax, cx, y_pos[i] - h/2, cx, y_pos[i+1] + h/2)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_3_Preprocessing.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# Figure 4: Consensus Model Architecture
# ==========================================
def fig4_consensus_arch():
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    w, h = 2.5, 1
    
    xs = [2, 6, 10]
    
    for x in xs:
        draw_box(ax, x, 9, w, h, "Descriptors")
        draw_arrow(ax, x, 9 - h/2, x, 6.5 + h/2)
        
    draw_box(ax, xs[0], 6.5, w, h, "RF")
    draw_box(ax, xs[1], 6.5, w, h, "XGBoost")
    draw_box(ax, xs[2], 6.5, w, h, "ASNN")
    
    for x in xs:
        draw_arrow(ax, x, 6.5 - h/2, x, 4 + h/2)
        draw_box(ax, x, 4, w, h, "Prediction")
        
    ax.plot([xs[0], xs[2]], [2.5, 2.5], color='black', lw=2)
    for x in xs:
        draw_arrow(ax, x, 4 - h/2, x, 2.5)
        
    draw_arrow(ax, 6, 2.5, 6, 1.5 + h/2)
    draw_box(ax, 6, 1.5, 3, h, "Average")
    
    draw_arrow(ax, 6, 1.5 - h/2, 6, -0.5 + h/2)
    draw_box(ax, 6, -0.5, 4, h, "Final Prediction", fontsize=14)
    
    ax.set_ylim(-2, 10.5)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_4_Consensus_Arch.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Figure 1: Workflow...")
    fig1_workflow()
    print("Generating Figure 2: Scaffold Split...")
    import numpy as np
    fig2_scaffold_split()
    print("Generating Figure 3: Preprocessing Pipeline...")
    fig3_preprocessing()
    print("Generating Figure 4: Consensus Architecture...")
    fig4_consensus_arch()
    print("Done generating schematic diagrams!")
