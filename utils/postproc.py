# utils/postproc.py
import numpy as np
from scipy import ndimage as ndi


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component in a 3-D binary mask."""
    labeled, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labeled, index=range(1, n + 1))
    max_label = (sizes.argmax() + 1).item()
    return (labeled == max_label).astype(mask.dtype)


def smooth_edges(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary morphological closing followed by opening."""
    struc = ndi.generate_binary_structure(3, 2)
    closed = ndi.binary_closing(mask, structure=struc, iterations=iterations)
    opened = ndi.binary_opening(closed, structure=struc, iterations=iterations)
    return opened.astype(mask.dtype)
