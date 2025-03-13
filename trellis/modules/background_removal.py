import torch
import numpy as np
from PIL import Image

class RembgBackgroundRemover:
    """Background removal using the rembg library
    
    This class provides a simple wrapper around rembg for background removal.
    rembg is a state-of-the-art background removal tool built on top of U2Net.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        # Check if rembg is available
        try:
            import rembg
            self.rembg_available = True
            self.rembg = rembg
        except ImportError:
            self.rembg_available = False
            print("Warning: rembg not installed. Please install it with: pip install rembg")
    
    def remove_background(self, image, alpha_matting=True, alpha_matting_foreground_threshold=240):
        """Remove background from an image using rembg
        
        Args:
            image (PIL.Image): Input image
            alpha_matting (bool): Whether to use alpha matting
            alpha_matting_foreground_threshold (int): Alpha matting foreground threshold
            
        Returns:
            PIL.Image: Image with background removed
            numpy.ndarray: Foreground mask
        """
        if not self.rembg_available:
            print("rembg not available. Returning original image.")
            # Return original image and a dummy mask
            width, height = image.size
            return image, np.ones((height, width), dtype=np.float32)
        
        # Convert PIL Image to bytes
        img_bytes = image.tobytes()
        
        # Process with rembg
        try:
            # Process with rembg
            output = self.rembg.remove(
                image,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold
            )
            
            # Extract alpha channel as mask
            if output.mode == 'RGBA':
                # Get alpha channel
                r, g, b, a = output.split()
                mask = np.array(a) / 255.0
                
                # Apply mask to original image
                image_array = np.array(image)
                for c in range(3):  # RGB channels
                    image_array[:, :, c] = image_array[:, :, c] * mask
                
                # Create masked image in RGB mode
                masked_image = Image.fromarray(image_array)
                
                return masked_image, mask
            else:
                # If output is not RGBA, return original image
                return output, np.ones((output.height, output.width), dtype=np.float32)
                
        except Exception as e:
            print(f"Error removing background with rembg: {e}")
            # Return original image and a dummy mask
            width, height = image.size
            return image, np.ones((height, width), dtype=np.float32)


class SimpleBackgroundRemover:
    """A simpler background remover using color thresholding.
    
    This is a fallback implementation when rembg is not available.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def remove_background(self, image, threshold=20, soft_mask=True):
        """Remove background using color thresholding
        
        This implementation assumes the object is centered and the background is relatively uniform.
        
        Args:
            image (PIL.Image): Input image
            threshold (int): Color difference threshold
            soft_mask (bool): Whether to use soft mask
            
        Returns:
            PIL.Image: Image with background removed
            numpy.ndarray: Foreground mask
        """
        # Convert to numpy array
        img_np = np.array(image)
        height, width, _ = img_np.shape
        
        # Assume background color is at the corners
        corners = [
            img_np[0, 0],      # Top-left
            img_np[0, width-1], # Top-right
            img_np[height-1, 0], # Bottom-left
            img_np[height-1, width-1] # Bottom-right
        ]
        
        # Average background color
        bg_color = np.mean(corners, axis=0)
        
        # Create mask based on color difference
        mask = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                # Color difference
                diff = np.sum(np.abs(img_np[y, x] - bg_color))
                # Normalize to 0-1
                if soft_mask:
                    mask[y, x] = min(1.0, diff / (3 * threshold))
                else:
                    mask[y, x] = 1.0 if diff > threshold else 0.0
        
        # Apply mask to image
        masked_img_np = img_np.copy()
        for c in range(3):
            masked_img_np[:, :, c] = masked_img_np[:, :, c] * mask
        
        # Convert back to PIL image
        masked_image = Image.fromarray(masked_img_np)
        
        return masked_image, mask


def create_background_remover(method='rembg', device='cuda'):
    """Factory function to create a background remover
    
    Args:
        method (str): Method to use ('rembg' or 'simple')
        device (str): Device to use
        
    Returns:
        RembgBackgroundRemover or SimpleBackgroundRemover: Background remover instance
    """
    if method == 'rembg':
        return RembgBackgroundRemover(device=device)
    else:
        return SimpleBackgroundRemover(device=device)
