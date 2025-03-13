import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trellis.pipelines.enhanced_multi_view_pipeline import EnhancedMultiViewPipeline
import cv2


def visualize_camera_positions(camera_params, save_path=None):
    """Visualize camera positions in 3D space
    
    Args:
        camera_params (List[Dict]): List of camera parameter dictionaries
        save_path (str, optional): Path to save the visualization
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera colors for different views
    colors = {'front': 'red', 'back': 'blue', 'left': 'green', 'right': 'orange'}
    
    # Plot each camera
    for i, params in enumerate(camera_params):
        # Extract camera position (translation vector)
        extrinsic = params['extrinsic']
        if isinstance(extrinsic, torch.Tensor):
            extrinsic = extrinsic.cpu().numpy()
        
        # Camera position is negative of translation vector in camera space
        # R * t = position in world space
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        camera_pos = -np.linalg.inv(rotation) @ translation
        
        # Camera orientation (simplified as a line from position toward lookAt point)
        view_type = params.get('view_type', f'camera_{i}')
        color = colors.get(view_type, 'gray')
        
        # Plot camera position
        ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], 
                  color=color, marker='o', s=100, label=view_type)
        
        # Plot camera orientation (forward direction)
        forward = -rotation[2, :]  # Third row of rotation matrix is backward in camera space
        ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                 forward[0], forward[1], forward[2], 
                 color=color, length=0.5, arrow_length_ratio=0.2)
    
    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions and Orientations')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    plt.legend()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Camera visualization saved to {save_path}")
    
    plt.tight_layout()
    return fig


def visualize_feature_matches(img1, img2, output_path=None):
    """Visualize feature matches between two images using SIFT"""
    # Convert PIL images to OpenCV format
    img1_cv = np.array(img1)[:, :, ::-1]  # RGB to BGR
    img2_cv = np.array(img2)[:, :, ::-1]
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1_cv, None)
    kp2, des2 = sift.detectAndCompute(img2_cv, None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Draw matches
    img_matches = cv2.drawMatches(img1_cv, kp1, img2_cv, kp2, good_matches[:30], None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Save if requested
    if output_path:
        cv2.imwrite(output_path, img_matches)
        print(f"Feature matches visualization saved to {output_path}")
    
    return img_matches


def run_multi_view_example(input_dir=None):
    """Run an example of the enhanced multi-view pipeline
    
    Args:
        input_dir (str, optional): Directory containing input images
            If not provided, will look for images in './input_images'
    """
    # Create output directory
    output_dir = "./multi_view_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize pipeline with enhanced models and SIFT feature detection
        pipeline = EnhancedMultiViewPipeline(
            use_rembg=True,                 # Use rembg for background removal
            use_enhanced_models=True,       # Use our enhanced flow models
            camera_feature_method='sift',   # Use SIFT for feature detection
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Set input directory
        if input_dir is None:
            input_dir = "./input_images"
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"Input directory {input_dir} does not exist. Creating it.")
            os.makedirs(input_dir, exist_ok=True)
            print(f"Please place your multi-view images in {input_dir}")
            print("Expected filenames: front.jpg, back.jpg, left.jpg, right.jpg")
            print("You can also use different filenames, but make sure they are in the correct order.")
            return None
        
        # Load user-provided images
        image_files = sorted([f for f in os.listdir(input_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        if not image_files:
            print(f"No images found in {input_dir}")
            print("Please add images to this directory and run again.")
            return None
        
        print(f"Found {len(image_files)} images in {input_dir}:")
        for f in image_files:
            print(f"  - {f}")
        
        # Load images
        images = []
        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                print(f"Loaded {img_path}, size: {img.size}")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if len(images) < 2:
            print("Need at least 2 images for multi-view reconstruction.")
            return None
        
        # Try to infer view types from filenames
        view_types = []
        for filename in image_files:
            base = os.path.splitext(filename.lower())[0]
            if 'front' in base:
                view_types.append('front')
            elif 'back' in base:
                view_types.append('back')
            elif 'left' in base:
                view_types.append('left')
            elif 'right' in base:
                view_types.append('right')
            elif 'top' in base:
                view_types.append('top')
            elif 'bottom' in base:
                view_types.append('bottom')
            else:
                view_types.append(f'view_{len(view_types)}')
        
        # Visualize feature matches between pairs of images
        if len(images) >= 2:
            print("\nVisualizing feature matches between views...")
            # Visualize between first image and all others
            for i in range(1, min(4, len(images))):
                matches_path = os.path.join(output_dir, f"matches_{0}_to_{i}.jpg")
                visualize_feature_matches(images[0], images[i], matches_path)
        
        # Run the pipeline with just gaussian representation (avoiding mesh which requires flexicubes)
        print("\nRunning enhanced multi-view pipeline...")
        try:
            results = pipeline.run_multi_view(
                images=images,
                view_types=view_types,
                formats=['gaussian']  # Request only Gaussian splat outputs to avoid mesh extraction issues
            )
        except ModuleNotFoundError as e:
            if "flexicubes" in str(e):
                print("Warning: FlexiCubes module is missing, using point cloud output instead.")
                # Try again with just point_cloud format
                results = pipeline.run_multi_view(
                    images=images,
                    view_types=view_types,
                    formats=['point_cloud']
                )
            else:
                raise
        
        # Save processed images with backgrounds removed
        print("\nSaving processed images...")
        for i, img in enumerate(results['processed_images']):
            img.save(os.path.join(output_dir, f"{view_types[i]}_processed.png"))
        
        # Visualize camera positions
        print("Visualizing camera positions...")
        fig = visualize_camera_positions(
            results['camera_params'],
            save_path=os.path.join(output_dir, "camera_positions.png")
        )
        
        # Save 3D outputs
        print("\nSaving 3D outputs...")
        
        # Try to save available formats
        if 'mesh' in results:
            try:
                mesh_output_path = os.path.join(output_dir, "output_mesh.obj")
                results['mesh'].export(mesh_output_path)
                print(f"Mesh saved to {mesh_output_path}")
            except Exception as e:
                print(f"Warning: Could not save mesh due to: {e}")
        
        if 'gaussian' in results:
            try:
                gaussian_output_path = os.path.join(output_dir, "output_gaussian.ply")
                results['gaussian'].export(gaussian_output_path)
                print(f"Gaussian splat saved to {gaussian_output_path}")
            except Exception as e:
                print(f"Warning: Could not save gaussian splat due to: {e}")
                
        if 'point_cloud' in results:
            try:
                pc_output_path = os.path.join(output_dir, "output_pointcloud.ply")
                results['point_cloud'].export(pc_output_path)
                print(f"Point cloud saved to {pc_output_path}")
            except Exception as e:
                print(f"Warning: Could not save point cloud due to: {e}")
        
        # Print completion message
        print("\nMulti-view 3D reconstruction complete!")
        print(f"All outputs saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"Error in multi-view example: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_multi_view_example(input_dir)
