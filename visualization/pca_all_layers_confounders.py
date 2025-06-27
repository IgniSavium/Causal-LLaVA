import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Input and output paths
FEATURE_PATH = '/path/to/Causal-LLaVA/confounders/output/visual_confounders.bin'
OUTPUT_ROOT = '/path/to/Causal-LLaVA/visualization/output/visual_confounders/'

# Highlighted IDs in MSCOCO
HIGHLIGHT_ID_GREEN = '67'   # dining table
HIGHLIGHT_ID_RED = ['1', '62', '47', '44', '51', '46', '49', '84', '48', '61']  # top-10 co-occurring objects in llava training data

# Load the weight file (PyTorch binary file)
weight_dict = torch.load(FEATURE_PATH)

# Check if the weight dictionary is empty
if not weight_dict:
    raise ValueError("The weight dictionary is empty.")

# Iterate through each key and its corresponding tensor in the weight dictionary
for key, tensor in weight_dict.items():
    print(f"Processing key: {key}")
    
    # Create subdirectory path for saving outputs
    output_dir = os.path.join(OUTPUT_ROOT, key.replace('.', '_'))  # Replace '.' with '_' to avoid path issues
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    # Convert tensor to NumPy array
    category_features = tensor.cpu().numpy()
    
    # Create a list of category IDs (assuming each row corresponds to a category)
    category_ids = list(map(str, range(category_features.shape[0])))
    
    # Data preprocessing: Standardize the features
    scaler = StandardScaler()
    category_features_scaled = scaler.fit_transform(category_features)
    
    # Apply PCA for dimensionality reduction, selecting the first three principal components
    pca = PCA(n_components=3)
    category_features_pca = pca.fit_transform(category_features_scaled)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set the title of the plot
    ax.set_title(f"3D PCA Visualization of {key}", fontsize=14)
    
    # Define colors for points: Red for highlighted IDs (except 67), Green for ID 67, Blue for others
    colors = ['green' if category_id == HIGHLIGHT_ID_GREEN 
              else ('red' if category_id in HIGHLIGHT_ID_RED else 'blue') 
              for category_id in category_ids]
    
    # Plot 3D scatter plot with color mapping
    scatter = ax.scatter(
        category_features_pca[:, 0],  # First principal component
        category_features_pca[:, 1],  # Second principal component
        category_features_pca[:, 2],  # Third principal component
        c=colors,  # Use the color list
        marker='o',  # Marker shape
        alpha=0.7,  # Transparency
        s=50  # Marker size
    )
    
    # Annotate red and green points with their category IDs
    for i, txt in enumerate(category_ids):
        if colors[i] in ['red', 'green']:  # Only annotate red and green points
            ax.text(
                category_features_pca[i, 0],  # x-coordinate
                category_features_pca[i, 1],  # y-coordinate
                category_features_pca[i, 2],  # z-coordinate
                txt,  # Annotation text
                fontsize=8,
                color='black'
            )
    
    # Set axis labels
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_zlabel("Principal Component 3", fontsize=12)
    
    # Loop through all combinations of elevation (elev) and azimuth (azim) angles
    for elev in range(0, 91, 10):  # Elevation from 0 to 180, step 10
        for azim in range(0, 91, 10):  # Azimuth from 0 to 90, step 10
            # Set the view angle for the 3D plot
            ax.view_init(elev=elev, azim=azim)
            
            # Generate filename with elev and azim values
            filename = f"3d_PCA_elev{elev}_azim{azim}.png"
            
            # Check if the file already exists
            if os.path.exists(os.path.join(output_dir, filename)):
                print(f"File {filename} already exists in {output_dir}.")
            else:
                # Save the 3D PCA plot as an image file
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                print(f"3D map saved successfully as {filename} in {output_dir}.")
    
    # Close the plot to free memory
    plt.close()