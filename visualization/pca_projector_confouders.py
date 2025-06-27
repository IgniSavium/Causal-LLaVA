import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.decomposition import PCA  # For dimensionality reduction
import os  # For file path operations

# Input and output paths
FEATURE_PATH = '/path/to/Causal-LLaVA/confounders/output/projector_confounders.bin'
OUTPUT_PATH = '/path/to/Causal-LLaVA/visualization/output/projector_confounders/'

# Highlighted IDs in MSCOCO
HIGHLIGHT_ID_GREEN = '67'   # dining table
HIGHLIGHT_ID_RED = ['1', '62', '47', '44', '51', '46', '49', '84', '48', '61']  # top-10 co-occurring objects in llava training data

# Load data from the binary file
weight_dict = torch.load(FEATURE_PATH)

# Extract the 'visual_confounders' weights from the loaded dictionary
if 'model.mm_projector.visual_confounders' not in weight_dict:
    raise KeyError("Key 'model.mm_projector.visual_confounders' not found in the loaded dictionary.")
visual_confounders = weight_dict['model.mm_projector.visual_confounders']

# Convert the weights to a NumPy array for processing
category_features = visual_confounders.cpu().numpy()

# Create a list of category IDs, assuming each row corresponds to one category
category_ids = list(map(str, range(category_features.shape[0])))

# Normalize the features using StandardScaler
scaler = StandardScaler()
category_features_scaled = scaler.fit_transform(category_features)

# Perform PCA to reduce dimensions to 3 principal components
pca = PCA(n_components=3)
category_features_pca = pca.fit_transform(category_features_scaled)

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Create a 3D plot for visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Set the title of the plot
ax.set_title("3D PCA Visualization of Category Features (Highlighted Points)", fontsize=14)

# Assign colors: red for HIGHLIGHT_ID_RED, green for HIGHLIGHT_ID_GREEN, and blue for others
colors = ['green' if category_id == HIGHLIGHT_ID_GREEN 
          else ('red' if category_id in HIGHLIGHT_ID_RED else 'blue') 
          for category_id in category_ids]

# Plot the 3D scatter plot with assigned colors
scatter = ax.scatter(
    category_features_pca[:, 0],  # First principal component
    category_features_pca[:, 1],  # Second principal component
    category_features_pca[:, 2],  # Third principal component
    c=colors,  # Color mapping
    marker='o',  # Marker style
    alpha=0.7,  # Transparency level
    s=50  # Marker size
)

# Annotate the highlighted points (red and green) with their category IDs
for i, txt in enumerate(category_ids):
    if colors[i] in ['red', 'green']:  # Only annotate red and green points
        ax.text(
            category_features_pca[i, 0],  # X-coordinate
            category_features_pca[i, 1],  # Y-coordinate
            category_features_pca[i, 2],  # Z-coordinate
            txt,  # Annotation text
            fontsize=8,
            color='black'
        )

# Label the axes
ax.set_xlabel("Principal Component 1", fontsize=12)
ax.set_ylabel("Principal Component 2", fontsize=12)
ax.set_zlabel("Principal Component 3", fontsize=12)

# Iterate over different elevation (elev) and azimuth (azim) angles for multiple views
for elev in range(0, 91, 10):  # Elevations from 0 to 90 degrees, step 10
    for azim in range(0, 91, 10):  # Azimuths from 0 to 90 degrees, step 10
        # Set the view angle
        ax.view_init(elev=elev, azim=azim)

        # Generate a unique filename for each view
        filename = f"3d_PCA_elev{elev}_azim{azim}.png"

        # Check if the file already exists to avoid overwriting
        if os.path.exists(os.path.join(OUTPUT_PATH, filename)):
            print(f"File {filename} already exists.")
        else:
            # Save the current view as a PNG file
            plt.savefig(os.path.join(OUTPUT_PATH, filename), dpi=300, bbox_inches='tight')
            print(f"3D map saved successfully as {filename}.")

# Close the plot to free resources
plt.close()