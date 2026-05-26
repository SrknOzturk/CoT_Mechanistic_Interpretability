"""
src/visualization.py
This module contains utility functions for visualizing tensors and activation 
patching results (e.g., heatmaps) using Matplotlib.
"""

import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union

def imshow(
    tensor: Union[torch.Tensor, np.ndarray, list],
    xaxis: str = "Head",
    yaxis: str = "Layer",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "seismic",
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> None:
    """
    Renders a 2D tensor or array as a heatmap using Matplotlib.
    Particularly useful for visualizing [n_layers, n_heads] activation patching scores.

    Args:
        tensor: The 2D data to plot (PyTorch Tensor, NumPy array, or list).
        xaxis: Label for the X-axis (default: "Head").
        yaxis: Label for the Y-axis (default: "Layer").
        title: Optional title for the plot.
        figsize: Tuple determining the size of the figure.
        cmap: Matplotlib colormap string.
        save_path: Directory path to save the generated figure.
        save_name: Specific filename for the figure. If None, it derives from the title.
        vmin: Minimum data value that corresponds to the lower end of the colormap.
        vmax: Maximum data value that corresponds to the upper end of the colormap.
    """

    # Convert to numpy if it's a torch tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif not isinstance(tensor, np.ndarray):
        tensor = np.array(tensor)

    # Create figure
    plt.figure(figsize=figsize)

    # Render heatmap with bounded scaling
    plt.imshow(tensor, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # Add UI elements
    plt.colorbar()
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    
    if title:
        plt.title(title)
        
    plt.tight_layout()

    # Save figure if a save_path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        # Determine a safe filename
        if save_name is not None:
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', save_name)
            safe_title = safe_title.replace('\n', '_').replace('\r', '_').strip().replace(' ', '_')[:200]
            if not safe_title:
                safe_title = "imshow_plot"
        elif title:
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        else:
            safe_title = "imshow_plot"
            
        file_path = os.path.join(save_path, f"{safe_title}.png")
        plt.savefig(file_path, bbox_inches='tight', dpi=200)
        print(f"✅ Figure saved to: {file_path}")

    # Display the figure
    plt.show()