import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional
from PIL import Image


class CameraPoseEstimator:
    """Camera pose estimation using feature matching and fundamental matrix estimation.
    
    This class provides methods to estimate relative camera poses from multiple views
    of an object, without assuming perfectly orthogonal views. It works by:    
    1. Detecting features in each image
    2. Matching features between pairs of images
    3. Computing the fundamental matrix between pairs
    4. Recovering relative rotation and translation
    5. Combining the relative poses into a global set of poses
    """
    
    def __init__(self, 
                 feature_method='sift',
                 K=None,  # Optional intrinsic matrix
                 min_matches=20,
                 device='cuda'):
        self.device = device
        self.feature_method = feature_method
        self.min_matches = min_matches
        
        # Initialize camera intrinsics with None (will be estimated if not provided)
        self.K = K
        
    def detect_features(self, image):
        """Detect features in an image using SIFT or ORB
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            Tuple[List, np.ndarray]: Keypoints and descriptors
        """
        if self.feature_method == 'sift':
            detector = cv2.SIFT_create()
        else:  # ORB as fallback
            detector = cv2.ORB_create(nfeatures=2000)
            
        keypoints, descriptors = detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Match features between two images
        
        Args:
            desc1 (np.ndarray): Descriptors from first image
            desc2 (np.ndarray): Descriptors from second image
            
        Returns:
            List[cv2.DMatch]: Matched features
        """
        if self.feature_method == 'sift':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(desc1, desc2, k=2)
            # Apply ratio test to filter good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        else:  # ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            good_matches = bf.match(desc1, desc2)
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            
        return good_matches[:100]  # Limit to top 100 matches
    
    def estimate_intrinsics(self, images):
        """Estimate camera intrinsic parameters from a set of images
        
        Args:
            images (List[np.ndarray]): List of input images
            
        Returns:
            np.ndarray: 3x3 camera intrinsic matrix
        """
        # If intrinsics are already provided, use them
        if self.K is not None:
            return self.K
            
        # Simple estimation based on image size
        # In a real implementation, would use a proper calibration method
        h, w = images[0].shape[:2]
        focal_length = max(w, h)
        center_x, center_y = w / 2, h / 2
        
        # Create camera intrinsic matrix
        K = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def compute_poses(self, images, pil_images=None, baseline_scale=1.0):
        """Compute camera poses from a set of images
        
        Args:
            images (List[np.ndarray]): List of input images (OpenCV format)
            pil_images (List[PIL.Image], optional): Original PIL images if available
            baseline_scale (float): Scale factor for the camera baseline
            
        Returns:
            List[Dict]: List of camera parameter dictionaries, each containing:
                - extrinsic: 4x4 extrinsic matrix (camera to world)
                - intrinsic: 3x3 intrinsic matrix
                - view_type: String identifier for the view
        """
        if len(images) < 2:
            raise ValueError("At least two images are required for pose estimation")
        
        # For convenience in naming views
        view_names = ['front', 'back', 'left', 'right', 'top', 'bottom'] + \
                     [f'view_{i}' for i in range(10)]
        
        # Estimate camera intrinsics
        K = self.estimate_intrinsics(images)
        
        # Detect features in all images
        all_keypoints = []
        all_descriptors = []
        for img in images:
            keypoints, descriptors = self.detect_features(img)
            all_keypoints.append(keypoints)
            all_descriptors.append(descriptors)
        
        # Initialize camera poses
        # First camera is the reference frame (identity matrix)
        R_list = [np.eye(3)]  # Rotation matrices
        t_list = [np.zeros((3, 1))]  # Translation vectors
        
        # Reference image is the first one
        ref_idx = 0
        ref_kp = all_keypoints[ref_idx]
        ref_desc = all_descriptors[ref_idx]
        
        # Compute relative poses for each image relative to the reference
        for i in range(len(images)):
            if i == ref_idx:
                continue  # Skip reference image
                
            # Match features
            matches = self.match_features(ref_desc, all_descriptors[i])
            
            # Check if we have enough matches
            if len(matches) < self.min_matches:
                print(f"Warning: Not enough matches between reference and image {i}")
                # Fall back to default pose
                if i == 1:  # Assume second image is looking from the opposite side
                    # 180-degree rotation around y-axis
                    R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                    t = np.array([[0, 0, baseline_scale]]).T
                elif i == 2:  # Third image from the left side
                    # 90-degree rotation around y-axis
                    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                    t = np.array([[-baseline_scale, 0, 0]]).T
                elif i == 3:  # Fourth image from the right side
                    # -90-degree rotation around y-axis
                    R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                    t = np.array([[baseline_scale, 0, 0]]).T
                else:
                    # Default to identity rotation and no translation
                    R = np.eye(3)
                    t = np.array([[0, 0, baseline_scale * i]]).T
            else:
                # Extract matched keypoints
                pts1 = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([all_keypoints[i][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Compute essential matrix
                E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                # Recover relative pose
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
                
                # Scale the translation to match the expected baseline
                t = t * baseline_scale / max(0.1, np.linalg.norm(t))
            
            # Add to pose lists
            R_list.append(R)
            t_list.append(t)
        
        # Convert relative poses to world poses
        camera_params = []
        for i in range(len(images)):
            # Create extrinsic matrix (camera to world)
            R = R_list[i]
            t = t_list[i]
            
            # Construct 4x4 extrinsic matrix
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3:4] = t
            
            # Determine view type
            view_type = view_names[i] if i < len(view_names) else f'view_{i}'
            
            # Create camera parameters dictionary
            camera_param = {
                'extrinsic': extrinsic,
                'intrinsic': K,
                'view_type': view_type,
                # Add image dimensions
                'height': images[i].shape[0],
                'width': images[i].shape[1]
            }
            
            camera_params.append(camera_param)
        
        return camera_params
    
    def estimate_poses(self, images):
        """Estimate camera poses from a list of PIL images
        
        Args:
            images (List[PIL.Image]): List of input images
            
        Returns:
            List[Dict]: List of camera parameter dictionaries
        """
        # Convert PIL images to OpenCV format
        cv_images = [np.array(img.convert('RGB'))[:, :, ::-1] for img in images]
        
        # Compute poses
        poses = self.compute_poses(cv_images, pil_images=images)
        
        return poses
    
    def visualize_matches(self, img1, img2, kp1, kp2, matches, path=None):
        """Visualize feature matches between two images
        
        Args:
            img1 (np.ndarray): First image
            img2 (np.ndarray): Second image
            kp1 (List): Keypoints from first image
            kp2 (List): Keypoints from second image
            matches (List): List of DMatch objects
            path (str, optional): Path to save visualization
            
        Returns:
            np.ndarray: Visualization image
        """
        # Draw matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Save if requested
        if path:
            cv2.imwrite(path, img_matches)
        
        return img_matches
