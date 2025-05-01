import torch
from torch.functional import F
import os
import numpy as np
import json
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from matplotlib.colors import TABLEAU_COLORS as colormap

from tqdm import tqdm
from contextlib import nullcontext

import comfy.model_management as mm
from comfy.utils import ProgressBar, common_upscale
import folder_paths

def sam2_segment_helper(image, sam2_model, keep_model_loaded, coordinates_positive=None, coordinates_negative=None, 
                individual_objects=False, bboxes=None, mask=None):
    """
    Helper function for SAM2 segmentation that handles both single image and video segmentation.
    
    Args:
        image: Input image tensor of shape (B, H, W, C)
        sam2_model: Dictionary containing the SAM2 model and its configuration
        keep_model_loaded: Whether to keep the model in GPU memory after processing
        coordinates_positive: JSON string of positive click coordinates
        coordinates_negative: JSON string of negative click coordinates
        individual_objects: If True, process each object separately
        bboxes: List of bounding boxes for guided segmentation
        mask: Optional input mask for refinement
    
    Returns:
        Tensor: Segmentation mask of shape (B, H, W)
    """
    offload_device = mm.unet_offload_device()
    model = sam2_model["model"]
    device = sam2_model["device"]
    dtype = sam2_model["dtype"]
    segmentor = sam2_model["segmentor"]
    B, H, W, C = image.shape

    # Handle input mask if provided
    if mask is not None:
        input_mask = mask.clone().unsqueeze(1)
        input_mask = F.interpolate(input_mask, size=(256, 256), mode="bilinear")
        input_mask = input_mask.squeeze(1)

    # Validate segmentor type and configuration
    if segmentor == 'automaskgenerator':
        raise ValueError("For automatic mask generation, use Sam2AutoMaskSegmentation node")
    if segmentor == 'single_image' and B > 1:
        print("Processing batch of images with single_image segmentor")
    if segmentor == 'video' and bboxes is not None and "2.1" not in sam2_model["version"]:
        raise ValueError("SAM2 2.0 does not support bounding boxes with video segmentor")

    # Resize input for video segmentation
    if segmentor == 'video':
        model_input_image_size = model.image_size
        print(f"Resizing input to {model_input_image_size}x{model_input_image_size}")
        image = common_upscale(image.movedim(-1,1), model_input_image_size, model_input_image_size, "bilinear", "disabled").movedim(1,-1)

    # Process point coordinates
    if coordinates_positive is not None:
        try:
            # Parse JSON coordinates and convert to point format
            coordinates_positive = json.loads(coordinates_positive.replace("'", '"'))
            coordinates_positive = [(coord['x'], coord['y']) for coord in coordinates_positive]
            if coordinates_negative is not None:
                coordinates_negative = json.loads(coordinates_negative.replace("'", '"'))
                coordinates_negative = [(coord['x'], coord['y']) for coord in coordinates_negative]
        except:
            pass
        
        # Format coordinates based on individual_objects setting
        if not individual_objects:
            positive_point_coords = np.atleast_2d(np.array(coordinates_positive))
        else:
            positive_point_coords = np.array([np.atleast_2d(coord) for coord in coordinates_positive])

        if coordinates_negative is not None:
            negative_point_coords = np.array(coordinates_negative)
            # Handle negative coordinates for individual objects mode
            if individual_objects:
                assert negative_point_coords.shape[0] <= positive_point_coords.shape[0], "Number of negative points cannot exceed positive points in individual objects mode"
                if negative_point_coords.ndim == 2:
                    negative_point_coords = negative_point_coords[:, np.newaxis, :]
                # Extend negative coordinates to match positive coordinates count
                while negative_point_coords.shape[0] < positive_point_coords.shape[0]:
                    negative_point_coords = np.concatenate((negative_point_coords, negative_point_coords[:1, :, :]), axis=0)
                final_coords = np.concatenate((positive_point_coords, negative_point_coords), axis=1)
            else:
                final_coords = np.concatenate((positive_point_coords, negative_point_coords), axis=0)
        else:
            final_coords = positive_point_coords

    # Process bounding boxes
    if bboxes is not None:
        boxes_np_batch = []
        for bbox_list in bboxes:
            boxes_np = []
            for bbox in bbox_list:
                boxes_np.append(bbox)
            boxes_np = np.array(boxes_np)
            boxes_np_batch.append(boxes_np)
        final_box = np.array(boxes_np_batch) if individual_objects else np.array(boxes_np)
        final_labels = None

    # Generate point labels
    if coordinates_positive is not None:
        if not individual_objects:
            positive_point_labels = np.ones(len(positive_point_coords))
        else:
            positive_labels = []
            for point in positive_point_coords:
                positive_labels.append(np.array([1]))
            positive_point_labels = np.stack(positive_labels, axis=0)
            
        if coordinates_negative is not None:
            if not individual_objects:
                negative_point_labels = np.zeros(len(negative_point_coords))
                final_labels = np.concatenate((positive_point_labels, negative_point_labels), axis=0)
            else:
                negative_labels = []
                for point in positive_point_coords:
                    negative_labels.append(np.array([0]))
                negative_point_labels = np.stack(negative_labels, axis=0)
                final_labels = np.concatenate((positive_point_labels, negative_point_labels), axis=1)                    
        else:
            final_labels = positive_point_labels
        print("Combined labels:", final_labels)
        print("Combined labels shape:", final_labels.shape)          

    # Initialize mask list and move model to device
    mask_list = []
    try:
        model.to(device)
    except:
        model.model.to(device)

    # Process with appropriate precision
    autocast_condition = not mm.is_device_mps(device)
    with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
        if segmentor == 'single_image':
            # Process single images
            image_np = (image.contiguous() * 255).byte().numpy()
            comfy_pbar = ProgressBar(len(image_np))
            tqdm_pbar = tqdm(total=len(image_np), desc="Processing Images")
            
            for i in range(len(image_np)):
                model.set_image(image_np[i])
                input_box = None if bboxes is None else (final_box[i] if len(image_np) > 1 else final_box)
                
                # Generate predictions
                out_masks, scores, logits = model.predict(
                    point_coords=final_coords if coordinates_positive is not None else None, 
                    point_labels=final_labels if coordinates_positive is not None else None,
                    box=input_box,
                    multimask_output=not individual_objects,
                    mask_input = input_mask[i].unsqueeze(0) if mask is not None else None,
                )
            
                # Process output masks
                if out_masks.ndim == 3:
                    # Sort and select best mask for single object mode
                    sorted_ind = np.argsort(scores)[::-1]
                    out_masks = out_masks[sorted_ind][0]
                    scores = scores[sorted_ind]
                    logits = logits[sorted_ind]
                    mask_list.append(np.expand_dims(out_masks, axis=0))
                else:
                    # Combine masks for multiple objects
                    _, _, H, W = out_masks.shape
                    combined_mask = np.zeros((H, W), dtype=bool)
                    for out_mask in out_masks:
                        combined_mask = np.logical_or(combined_mask, out_mask)
                    mask_list.append(combined_mask.astype(np.uint8))
                
                comfy_pbar.update(1)
                tqdm_pbar.update(1)

        elif segmentor == 'video':
            # Process video frames
            mask_list = []
            if hasattr(self, 'inference_state'):
                model.reset_state(self.inference_state)
            self.inference_state = model.init_state(image.permute(0, 3, 1, 2).contiguous(), H, W, device=device)
            
            input_box = None if bboxes is None else bboxes[0]
            
            if individual_objects and bboxes is not None:
                raise ValueError("Bounding boxes are not supported with individual objects in video mode")

            # Add points for tracking
            if individual_objects:
                for i, (coord, label) in enumerate(zip(final_coords, final_labels)):
                    _, out_obj_ids, out_mask_logits = model.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=i,
                        points=final_coords[i],
                        labels=final_labels[i],
                        clear_old_points=True,
                        box=input_box
                    )
            else:
                _, out_obj_ids, out_mask_logits = model.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=final_coords if coordinates_positive is not None else None, 
                    labels=final_labels if coordinates_positive is not None else None,
                    clear_old_points=True,
                    box=input_box
                )

            # Process video frames
            pbar = ProgressBar(B)
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(self.inference_state):
                if individual_objects:
                    # Combine masks for all objects in the frame
                    _, _, H, W = out_mask_logits.shape
                    combined_mask = np.zeros((H, W), dtype=np.uint8)
                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                        combined_mask = np.logical_or(combined_mask, out_mask)
                    video_segments[out_frame_idx] = combined_mask
                else:
                    # Store individual object masks
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                pbar.update(1)

            # Collect masks from video segments
            if individual_objects:
                mask_list.extend(video_segments.values())
            else:
                for obj_masks in video_segments.values():
                    mask_list.extend(obj_masks.values())

    # Offload model if requested
    if not keep_model_loaded:
        try:
            model.to(offload_device)
        except:
            model.model.to(offload_device)
    
    # Convert masks to tensor format
    out_list = []
    for mask in mask_list:
        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.permute(1, 2, 0)
        mask_tensor = mask_tensor[:, :, 0]
        out_list.append(mask_tensor)
    
    # Stack and return final mask tensor
    return torch.stack(out_list, dim=0).cpu().float()

class Sam2TiledSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input parameters for the SAM2 tiled segmentation node.
        """
        return {
            "required": {
                "sam2_model": ("SAM2MODEL", ),
                "image": ("IMAGE", ),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 64}),
                "tile_overlap": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "mask_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "mask_color": ("STRING", {
                    "default": "(255,0,0)", 
                    "multiline": False, 
                    "description": "Mask color in RGB format (r,g,b) with values 0-255"
                }),
            },
            "optional": {
                "coordinates_positive": ("STRING", {"forceInput": True}),
                "coordinates_negative": ("STRING", {"forceInput": True}),
                "bboxes": ("BBOX", ),
                "individual_objects": ("BOOLEAN", {"default": False}),
                "mask": ("MASK", ),
            },
        }
    
    RETURN_TYPES = ("MASK", "IMAGE", "BBOX", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mask", "tiles", "tile_bboxes", "annotated_image", "masked_tiles")
    FUNCTION = "segment"
    CATEGORY = "SAM2"

    def segment(self, sam2_model, image, tile_size, tile_overlap, keep_model_loaded, mask_opacity, mask_color,
                coordinates_positive=None, coordinates_negative=None, bboxes=None, 
                individual_objects=False, mask=None):
        try:
            from sahi.slicing import slice_image
            from sahi.utils.cv import read_image_as_pil
            from sahi.utils.coco import Coco, CocoImage, CocoAnnotation
            from sahi.utils.file import save_json
        except ImportError:
            raise ImportError("SAHI is not installed. Please install it with: pip install sahi")

        # Create a list of distinct colors for adjusted bounding boxes
        distinct_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
        # If more colors are needed, generate additional random colors
        while len(distinct_colors) < 100:  # Increase if necessary
            color = '#{:06x}'.format(random.randint(0, 0xFFFFFF))
            if color not in distinct_colors:
                distinct_colors.append(color)

        print(f"Input image shape: {image.shape}")
        print(f"Coordinates positive: {coordinates_positive}")
        print(f"Coordinates negative: {coordinates_negative}")
        print(f"Bboxes: {bboxes}")

        # Convert the ComfyUI image to PIL format
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        
        # Calculate tile dimensions
        width, height = image_pil.size
        slice_height = tile_size
        slice_width = tile_size
        overlap_height_ratio = tile_overlap
        overlap_width_ratio = tile_overlap

        print(f"Original image size: {width}x{height}")
        print(f"Tile size: {tile_size}x{tile_size}")
        print(f"Overlap ratio: {tile_overlap}")

        # Create figure for visualization
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(image_pil)

        # Divide image into tiles using SAHI
        slice_image_result = slice_image(
            image=image_pil,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

        print(f"Number of tiles: {len(slice_image_result.images)}")

        # Initialize final mask, tile batch and tile bboxes list
        final_mask = torch.zeros((height, width), dtype=torch.float32)
        tiles_batch = []
        tile_bboxes = []
        masked_tiles = []

        # Process each tile
        for slice_idx, slice_image in enumerate(slice_image_result.images):
            try:
                print(f"\nProcessing tile {slice_idx + 1}/{len(slice_image_result.images)}")
                print(f"Tile shape: {slice_image.shape}")

                # Convert tile to ComfyUI format
                slice_tensor = torch.from_numpy(np.array(slice_image)).float() / 255.0
                slice_tensor = slice_tensor.unsqueeze(0)
                print(f"Slice tensor shape: {slice_tensor.shape}")

                # Add tile to batch
                tiles_batch.append(slice_tensor)

                # Get tile coordinates
                starting_pixel = slice_image_result.starting_pixels[slice_idx]
                slice_height = slice_image.shape[0]
                slice_width = slice_image.shape[1]

                print(f"Tile position: {starting_pixel}")
                print(f"Tile dimensions: {slice_width}x{slice_height}")

                # Calculate final tile coordinates
                end_x = min(starting_pixel[0] + slice_width, width)
                end_y = min(starting_pixel[1] + slice_height, height)
                
                # Verify dimensions are valid
                if end_x <= starting_pixel[0] or end_y <= starting_pixel[1]:
                    print(f"Skipping tile {slice_idx} - Invalid dimensions")
                    continue

                # Add tile bounding box to bboxes list
                tile_bbox = [
                    starting_pixel[0],  # x1
                    starting_pixel[1],  # y1
                    end_x,              # x2
                    end_y               # y2
                ]
                tile_bboxes.append(tile_bbox)

                # Adjust bboxes for tile
                tile_bboxes_input = None
                if bboxes is not None:
                    print(f"\n{'='*50}")
                    print(f"Processing tile {slice_idx}")
                    print(f"Original bboxes: {bboxes}")
                    print(f"Tile position: ({starting_pixel[0]}, {starting_pixel[1]}) -> ({end_x}, {end_y})")
                    
                    tile_bboxes_list = []
                    for bbox_idx, bbox in enumerate(bboxes):
                        x1, y1, x2, y2 = bbox
                        # Verify if bbox intersects tile
                        if (x1 < end_x and x2 > starting_pixel[0] and
                            y1 < end_y and y2 > starting_pixel[1]):
                            # Calculate intersection
                            tile_x1 = max(x1 - starting_pixel[0], 0)
                            tile_y1 = max(y1 - starting_pixel[1], 0)
                            tile_x2 = min(x2 - starting_pixel[0], slice_width)
                            tile_y2 = min(y2 - starting_pixel[1], slice_height)
                            
                            if tile_x2 > tile_x1 and tile_y2 > tile_y1:  # Verify bbox is valid
                                print(f"BBox {bbox_idx} intersects tile:")
                                print(f"  Original: ({x1}, {y1}) -> ({x2}, {y2})")
                                print(f"  Adjusted: ({tile_x1}, {tile_y1}) -> ({tile_x2}, {tile_y2})")
                                tile_bboxes_list.append([tile_x1, tile_y1, tile_x2, tile_y2])

                                # Draw original bbox on image
                                rect = patches.Rectangle(
                                    (x1, y1),
                                    x2 - x1,
                                    y2 - y1,
                                    linewidth=2,
                                    edgecolor='red',
                                    facecolor='none',
                                    label=f'Object {bbox_idx}'
                                )
                                ax.add_patch(rect)

                                # Add object label
                                ax.text(x1, y1 - 5,  # Move 5 pixels up
                                       f' Object {bbox_idx} ',
                                       color='red',
                                       fontsize=8,
                                       bbox=dict(
                                           facecolor='white',
                                           alpha=0.7,
                                           edgecolor='none',
                                           pad=0.3,
                                           boxstyle='square'
                                       ),
                                       horizontalalignment='left',
                                       verticalalignment='bottom')

                                # Draw tile containing bbox
                                tile_rect = patches.Rectangle(
                                    (starting_pixel[0], starting_pixel[1]),
                                    end_x - starting_pixel[0],
                                    end_y - starting_pixel[1],
                                    linewidth=2,
                                    edgecolor=distinct_colors[slice_idx % len(distinct_colors)],
                                    facecolor='none',
                                    alpha=0.5,
                                    label=f'Tile {slice_idx}'
                                )
                                ax.add_patch(tile_rect)

                                # Add tile label
                                ax.text(starting_pixel[0], starting_pixel[1] - 5,  # Move 5 pixels up
                                       f' Tile {slice_idx} ',
                                       color=distinct_colors[slice_idx % len(distinct_colors)],
                                       fontsize=8,
                                       bbox=dict(
                                           facecolor='white',
                                           alpha=0.7,
                                           edgecolor='none',
                                           pad=0.3,
                                           boxstyle='square'
                                       ),
                                       horizontalalignment='left',
                                       verticalalignment='bottom')

                    if tile_bboxes_list:
                        tile_bboxes_input = tile_bboxes_list
                        print(f"Valid tile bboxes: {tile_bboxes_input}")
                    else:
                        print("No valid bboxes for this tile")

                # Adjust positive and negative coordinates for tile
                tile_coords_positive = None
                tile_coords_negative = None
                
                if coordinates_positive:
                    coords = json.loads(coordinates_positive)
                    print(f"Original positive coordinates: {coords}")
                    tile_coords = []
                    for coord in coords:
                        x, y = coord['x'], coord['y']
                        # Verify if point is inside tile
                        if (starting_pixel[0] <= x < end_x and
                            starting_pixel[1] <= y < end_y):
                            # Convert coordinates relative to tile
                            tile_coords.append({
                                'x': x - starting_pixel[0],
                                'y': y - starting_pixel[1]
                            })
                    if tile_coords:
                        tile_coords_positive = json.dumps(tile_coords)
                        print(f"Tile positive coordinates: {tile_coords_positive}")

                if coordinates_negative:
                    coords = json.loads(coordinates_negative)
                    print(f"Original negative coordinates: {coords}")
                    tile_coords = []
                    for coord in coords:
                        x, y = coord['x'], coord['y']
                        if (starting_pixel[0] <= x < end_x and
                            starting_pixel[1] <= y < end_y):
                            tile_coords.append({
                                'x': x - starting_pixel[0],
                                'y': y - starting_pixel[1]
                            })
                    if tile_coords:
                        tile_coords_negative = json.dumps(tile_coords)
                        print(f"Tile negative coordinates: {tile_coords_negative}")

                # Execute segmentation only if there are valid bounding boxes for this tile
                if tile_bboxes_input:
                    print(f"Starting segmentation for tile {slice_idx}")
                    try:
                        # Process all bounding boxes together
                        mask_result = sam2_segment_helper(
                            image=slice_tensor,
                            sam2_model=sam2_model,
                            keep_model_loaded=True,
                            coordinates_positive=tile_coords_positive,
                            coordinates_negative=tile_coords_negative,
                            bboxes=tile_bboxes_input,
                            individual_objects=True,
                            mask=None  # Temporarily remove input mask causing issues
                        )

                        print(f"Mask result shape: {mask_result.shape}")
                        print(f"Mask result type: {mask_result.dtype}")
                        print(f"Mask result range: [{mask_result.min()}, {mask_result.max()}]")
                        
                        # Handle resulting masks
                        tile_masks = []
                        if mask_result.dim() == 3:
                            for i in range(mask_result.shape[0]):
                                mask = mask_result[i]
                                print(f"Processing mask {i} with shape {mask.shape}")
                                if mask.sum() > 0:  # Verify mask is not empty
                                    tile_masks.append(mask)
                                    print(f"Added non-empty mask {i} with sum {mask.sum()}")
                        else:
                            if mask_result.sum() > 0:
                                tile_masks.append(mask_result)
                                print(f"Added single non-empty mask with sum {mask_result.sum()}")

                        # Combine tile masks
                        if tile_masks:
                            print(f"Combining {len(tile_masks)} masks for tile {slice_idx}")
                            # Initialize tile mask with zeros
                            tile_mask = torch.zeros_like(tile_masks[0], dtype=torch.float32)
                            
                            # Combine all masks using OR logical operation
                            for i, mask in enumerate(tile_masks):
                                print(f"Adding mask {i} with sum: {mask.sum()}")
                                tile_mask = torch.logical_or(tile_mask, mask)
                            
                            tile_mask = tile_mask.float()
                            print(f"Combined mask sum: {tile_mask.sum()}")

                            # Calculate actual dimensions for current tile
                            actual_height = min(slice_height, end_y - starting_pixel[1])
                            actual_width = min(slice_width, end_x - starting_pixel[0])

                            # Resize tile mask if necessary
                            if tile_mask.shape != (actual_height, actual_width):
                                print(f"Resizing mask from {tile_mask.shape} to {(actual_height, actual_width)}")
                                tile_mask = tile_mask[:actual_height, :actual_width]

                            # Update final mask
                            print(f"Updating final mask at [{starting_pixel[1]}:{end_y}, {starting_pixel[0]}:{end_x}]")
                            print(f"Current region shape: {final_mask[starting_pixel[1]:end_y, starting_pixel[0]:end_x].shape}")
                            print(f"Tile mask shape: {tile_mask.shape}")
                            
                            # Verify dimensions match
                            if final_mask[starting_pixel[1]:end_y, starting_pixel[0]:end_x].shape == tile_mask.shape:
                                final_mask[starting_pixel[1]:end_y, starting_pixel[0]:end_x] = torch.logical_or(
                                    final_mask[starting_pixel[1]:end_y, starting_pixel[0]:end_x],
                                    tile_mask
                                ).float()
                                print(f"Successfully updated final mask for tile {slice_idx}")
                            else:
                                print(f"Warning: Shape mismatch for tile {slice_idx}")
                                print(f"Final mask region shape: {final_mask[starting_pixel[1]:end_y, starting_pixel[0]:end_x].shape}")
                                print(f"Tile mask shape: {tile_mask.shape}")

                            # Visualize mask
                            try:
                                color_str = mask_color.replace('(', '').replace(')', '').replace(' ', '')
                                color_list = [int(x) for x in color_str.split(',')]
                                if len(color_list) != 3:
                                    raise ValueError("Mask must have 3 RGB components")
                                color_list = np.clip(color_list, 0, 255)
                            except Exception as e:
                                print(f"Error parsing color: {str(e)}. Using default red color.")
                                color_list = [255, 0, 0]

                            # Create colored mask
                            colored_mask = torch.zeros_like(slice_tensor)
                            colored_mask[..., 0] = color_list[0] / 255.0
                            colored_mask[..., 1] = color_list[1] / 255.0
                            colored_mask[..., 2] = color_list[2] / 255.0

                            # Apply opacity and combine with tile image
                            mask_overlay = tile_mask.unsqueeze(-1).expand(-1, -1, 3) * mask_opacity
                            masked_tile = slice_tensor * (1 - mask_overlay) + colored_mask * mask_overlay
                            masked_tiles.append(masked_tile)
                            print(f"Added masked tile with shape {masked_tile.shape}")
                        else:
                            print(f"No valid masks for tile {slice_idx}")
                            masked_tiles.append(slice_tensor)
                    except Exception as e:
                        print(f"Error processing tile {slice_idx}: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        masked_tiles.append(slice_tensor)
                else:
                    print(f"No bounding boxes for tile {slice_idx}")
                    masked_tiles.append(slice_tensor)

            except Exception as e:
                print(f"Error processing tile {slice_idx}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                masked_tiles.append(slice_tensor)
                continue

        # Remove axes and padding around image
        ax.axis('off')
        ax.margins(0,0)
        ax.get_xaxis().set_major_locator(plt.NullLocator())
        ax.get_yaxis().set_major_locator(plt.NullLocator())
        
        # Save annotated image
        fig.canvas.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', pad_inches=0)
        buf.seek(0)
        annotated_image_pil = Image.open(buf)
        plt.close(fig)

        # Convert annotated image to tensor
        annotated_image_tensor = torch.from_numpy(np.array(annotated_image_pil)[:, :, :3]).float() / 255.0
        annotated_image_tensor = annotated_image_tensor.unsqueeze(0)

        # Concatenate all tiles into a single batch
        if tiles_batch:
            tiles_tensor = torch.cat(tiles_batch, dim=0)
        else:
            tiles_tensor = torch.zeros((1, height, width, 3), dtype=torch.float32)

        # Convert tile bboxes list to tensor
        tile_bboxes_tensor = torch.tensor(tile_bboxes, dtype=torch.float32) if tile_bboxes else torch.zeros((1, 4), dtype=torch.float32)

        # Concatenate all masked tiles into a single batch
        if masked_tiles:
            masked_tiles_tensor = torch.cat(masked_tiles, dim=0)
        else:
            masked_tiles_tensor = torch.zeros((1, height, width, 3), dtype=torch.float32)

        print("\nFinal results:")
        print(f"Final mask shape: {final_mask.shape}")
        print(f"Final mask values range: [{final_mask.min()}, {final_mask.max()}]")
        print(f"Tiles tensor shape: {tiles_tensor.shape}")
        print(f"Tile bboxes shape: {tile_bboxes_tensor.shape}")
        print(f"Annotated image shape: {annotated_image_tensor.shape}")
        print(f"Masked tiles shape: {masked_tiles_tensor.shape}")

        return (final_mask, tiles_tensor, tile_bboxes_tensor, annotated_image_tensor, masked_tiles_tensor,)
     
class Sam2ContextSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input parameters for the SAM2 bounding box tiled segmentation node.
        Provides advanced options for mask processing and visualization.
        """
        return {
            "required": {
                "sam2_model": ("SAM2MODEL", ),
                "image": ("IMAGE", ),
                "context_scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.05,
                    "description": "Scale factor for context around bounding boxes"
                }),
                "force_square_context": ("BOOLEAN", {
                    "default": False,
                    "description": "Force context to be square using the longest side"
                }),
                "limit_tile_size": ("BOOLEAN", {
                    "default": True,
                    "description": "Enable/disable maximum tile size limit"
                }),
                "max_tile_size": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 128,
                    "description": "Maximum tile size (when limit is enabled)"
                }),
                "mask_filter_mode": (["disabled", "absolute", "percentage"], {
                    "default": "disabled",
                    "description": "Method to filter out small mask components"
                }),
                "min_mask_area": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 10000,
                    "step": 10,
                    "description": "Minimum area in pixels for a mask component"
                }),
                "min_mask_area_percent": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0001,
                    "max": 1.0,
                    "step": 0.001,
                    "description": "Minimum area as percentage of tile area"
                }),
                "fill_individual_masks": ("BOOLEAN", {
                    "default": False,
                    "description": "Fill holes in individual masks before combining"
                }),
                "close_mask_gaps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "description": "Connect mask parts that are within this many pixels of each other"
                }),
                "dilate_masks": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "description": "Number of pixels to expand the masks"
                }),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "mask_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "coordinates_positive": ("STRING", {"forceInput": True}),
                "coordinates_negative": ("STRING", {"forceInput": True}),
                "bboxes": ("BBOX", ),
                "individual_objects": ("BOOLEAN", {"default": False}),
                "mask": ("MASK", ),
            },
        }
    
    RETURN_TYPES = ("MASK", "BBOX", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mask", "tile_bboxes", "annotated_image", "removed_components", "colored_masks")
    FUNCTION = "segment"
    CATEGORY = "SAM2"

    def calculate_context_tile(self, bbox, context_scale, image_size, max_tile_size, force_square_context, limit_tile_size):
        # Calculate bbox center
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Calculate bbox dimensions
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if force_square_context:
            # Use the longer side as reference to create a square tile
            base_size = max(width, height)
            context_size = base_size * context_scale if not limit_tile_size else min(base_size * context_scale, max_tile_size)
            half_size = context_size / 2
            
            tile_x1 = max(0, center_x - half_size)
            tile_y1 = max(0, center_y - half_size)
            tile_x2 = min(image_size[1], center_x + half_size)
            tile_y2 = min(image_size[0], center_y + half_size)
        else:
            # Maintain original bbox proportions
            context_width = width * context_scale if not limit_tile_size else min(width * context_scale, max_tile_size)
            context_height = height * context_scale if not limit_tile_size else min(height * context_scale, max_tile_size)
            
            half_width = context_width / 2
            half_height = context_height / 2
            
            tile_x1 = max(0, center_x - half_width)
            tile_y1 = max(0, center_y - half_height)
            tile_x2 = min(image_size[1], center_x + half_width)
            tile_y2 = min(image_size[0], center_y + half_height)
        
        return [int(tile_x1), int(tile_y1), int(tile_x2), int(tile_y2)]

    def filter_small_components(self, mask, tile_area, min_mask_area, min_mask_area_percent, mode="absolute"):
        """
        Filters small components from the mask.
        Returns the filtered mask and a mask of removed components.
        """
        import cv2
        import numpy as np

        # If mask is 3D (batch, height, width), take the first mask
        if len(mask.shape) == 3:
            mask = mask[0]  # Now we have a 2D mask

        # Convert mask to uint8 for cv2
        mask_uint8 = (mask.cpu().numpy() * 255).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask_uint8)
        
        # Mask for components to keep and those removed
        kept_mask = np.zeros_like(mask_uint8)
        removed_mask = np.zeros_like(mask_uint8)
        
        # Calculate threshold based on mode
        if mode == "percentage":
            threshold = tile_area * min_mask_area_percent
        else:  # absolute
            threshold = min_mask_area
        
        # Analyze each component
        for label in range(1, num_labels):  # 0 is background
            component = (labels == label)
            area = component.sum()
            
            if area >= threshold:
                kept_mask[component] = 255
            else:
                removed_mask[component] = 255
        
        # Convert to tensors and handle batch case
        if len(mask.shape) == 3:
            kept_mask = torch.from_numpy(kept_mask > 0).float().unsqueeze(0)
            removed_mask = torch.from_numpy(removed_mask > 0).float().unsqueeze(0)
        else:
            kept_mask = torch.from_numpy(kept_mask > 0).float()
            removed_mask = torch.from_numpy(removed_mask > 0).float()
        
        return kept_mask, removed_mask

    def fill_mask_holes(self, mask):
        """
        Fills holes in masks using cv2.floodFill.
        Handles both 2D and 3D masks (batch, height, width).
        """
        import cv2
        import numpy as np
        
        # If mask is 3D (batch, height, width), take the first mask
        if len(mask.shape) == 3:
            mask = mask[0]  # Now we have a 2D mask
        
        # Convert mask to uint8
        mask_uint8 = (mask.cpu().numpy() * 255).astype(np.uint8)
        
        # Now we can get height and width
        h, w = mask_uint8.shape
        
        # Create a larger mask for flood fill
        padded = np.pad(mask_uint8, 1, mode='constant')
        
        # Create a mask for flood fill
        flood_mask = np.zeros((h+4, w+4), np.uint8)
        
        # Perform flood fill from borders
        cv2.floodFill(padded, flood_mask, (0,0), 255)
        
        # Invert the result
        filled = 255 - padded[1:-1, 1:-1]
        
        # Combine with original mask
        result = np.maximum(mask_uint8, filled)
        
        # Convert to tensor and add batch dimension if necessary
        result_tensor = torch.from_numpy(result > 0).float()
        if len(mask.shape) == 3:
            result_tensor = result_tensor.unsqueeze(0)
        
        return result_tensor

    def dilate_mask(self, mask, dilate_pixels):
        """
        Dilates the mask by the specified number of pixels.
        """
        if dilate_pixels <= 0:
            return mask

        import cv2
        import numpy as np

        # If mask is 3D (batch, height, width), take the first mask
        if len(mask.shape) == 3:
            mask = mask[0]

        # Convert mask to uint8 for cv2
        mask_uint8 = (mask.cpu().numpy() * 255).astype(np.uint8)

        # Create dilation kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_pixels + 1, 2 * dilate_pixels + 1))
        
        # Apply dilation
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)

        # Convert to tensor and handle batch case
        if len(mask.shape) == 3:
            dilated_mask = torch.from_numpy(dilated_mask > 0).float().unsqueeze(0)
        else:
            dilated_mask = torch.from_numpy(dilated_mask > 0).float()

        return dilated_mask

    def close_mask_gaps(self, mask, gap_size):
        """
        Closes gaps between parts of the mask that are within a certain distance.
        Uses a morphological closing operation (dilation followed by erosion).
        """
        if gap_size <= 0:
            return mask

        import cv2
        import numpy as np

        # If mask is 3D (batch, height, width), take the first mask
        if len(mask.shape) == 3:
            mask = mask[0]

        # Convert mask to uint8 for cv2
        mask_uint8 = (mask.cpu().numpy() * 255).astype(np.uint8)

        # Create kernel for closing operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_size + 1, 2 * gap_size + 1))
        
        # Apply closing operation (dilate followed by erode)
        closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        # Convert to tensor and handle batch case
        if len(mask.shape) == 3:
            closed_mask = torch.from_numpy(closed_mask > 0).float().unsqueeze(0)
        else:
            closed_mask = torch.from_numpy(closed_mask > 0).float()

        return closed_mask

    def segment(self, image, sam2_model, context_scale, force_square_context,
                limit_tile_size, max_tile_size, mask_filter_mode, min_mask_area, min_mask_area_percent,
                fill_individual_masks, close_mask_gaps, dilate_masks, keep_model_loaded, mask_opacity,
                coordinates_positive=None, coordinates_negative=None, bboxes=None,
                individual_objects=False, mask=None):
        
        print(f"DEBUG: Starting segmentation with {len(bboxes) if bboxes else 0} bounding boxes")
        print(f"DEBUG: Image dimensions: {image.shape}")
        print(f"DEBUG: Tile size limit {'enabled' if limit_tile_size else 'disabled'}")
        if limit_tile_size:
            print(f"DEBUG: Maximum tile size: {max_tile_size}")

        if bboxes is None or len(bboxes) == 0:
            print("No bounding boxes provided")
            return (torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32),
                    torch.zeros((1, 4), dtype=torch.float32),
                    torch.zeros_like(image),
                    torch.zeros_like(image),
                    torch.zeros_like(image))

        # Initialize final mask and removed components visualization
        final_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
        removed_components_vis = torch.zeros((1, image.shape[1], image.shape[2], 3), dtype=torch.float32)
        
        # Initialize image for colored masks
        colored_masks = torch.zeros((1, image.shape[1], image.shape[2], 3), dtype=torch.float32)

        # Generate distinct colors for masks
        def generate_distinct_colors(n):
            colors = []
            for i in range(n):
                # Generate random colors but avoid too dark or too light colors
                while True:
                    # Generate random RGB
                    color = [random.random() for _ in range(3)]
                    # Calculate brightness (approximate formula)
                    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                    # Ensure color is not too dark or too light
                    if 0.2 < brightness < 0.8:
                        # Increase saturation
                        max_val = max(color)
                        if max_val > 0:
                            color = [c/max_val for c in color]
                        colors.append(color)
                        break
            return colors

        # Generate colors for masks
        colors = generate_distinct_colors(len(bboxes))

        # Distinct colors for bounding boxes (keep this part separate)
        distinct_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        # Create figure for visualization of bboxes and tile
        width, height = image.shape[2], image.shape[1]
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow((image[0].cpu().numpy() * 255).astype(np.uint8))

        # Process each bbox individually
        for bbox_idx, bbox in enumerate(bboxes):
            print(f"\nProcessing bbox {bbox_idx}: {bbox}")
            
            # Calculate tile for this specific bbox
            tile_bbox = self.calculate_context_tile(
                bbox, context_scale, image.shape[1:], max_tile_size, force_square_context, limit_tile_size
            )
            x1, y1, x2, y2 = tile_bbox
            
            # Extract tile from image
            tile_image = image[0, y1:y2, x1:x2]
            tile_tensor = torch.from_numpy(np.array(tile_image)).float()
            tile_tensor = tile_tensor.unsqueeze(0)

            # Adjust bbox to tile coordinates
            tile_bbox_x1 = bbox[0] - x1
            tile_bbox_y1 = bbox[1] - y1
            tile_bbox_x2 = bbox[2] - x1
            tile_bbox_y2 = bbox[3] - y1
            tile_bbox_adjusted = [tile_bbox_x1, tile_bbox_y1, tile_bbox_x2, tile_bbox_y2]

            print(f"Processing tile shape: {tile_tensor.shape}")
            print(f"Adjusted bbox: {tile_bbox_adjusted}")

            try:
                # Execute segmentation ONLY for this bbox
                mask_result = sam2_segment_helper(
                    image=tile_tensor,
                    sam2_model=sam2_model,
                    keep_model_loaded=True,
                    coordinates_positive=coordinates_positive,
                    coordinates_negative=coordinates_negative,
                    bboxes=[tile_bbox_adjusted],  # Only the current bbox
                    individual_objects=True,
                    mask=None
                )

                print(f"Mask result shape: {mask_result.shape}")
                    
                # Process mask
                if mask_result.sum() > 0:

                    # If requested, fill holes in mask
                    if fill_individual_masks:
                        print(f"Filling holes in mask for bbox {bbox_idx}")
                        pre_fill_sum = mask_result.sum()
                        mask_result = self.fill_mask_holes(mask_result)
                        post_fill_sum = mask_result.sum()
                        print(f"Mask sum before fill: {pre_fill_sum}, after fill: {post_fill_sum}")

                    # Close gaps in mask if requested
                    if close_mask_gaps > 0:
                        print(f"Closing gaps of {close_mask_gaps} pixels in mask")
                        pre_close_sum = mask_result.sum()
                        mask_result = self.close_mask_gaps(mask_result, close_mask_gaps)
                        post_close_sum = mask_result.sum()
                        print(f"Mask sum before closing: {pre_close_sum}, after closing: {post_close_sum}")

                    # Filter small components if requested
                    if mask_filter_mode != "disabled":
                        tile_area = (y2 - y1) * (x2 - x1)
                        mask_result, removed_components = self.filter_small_components(
                            mask_result,
                            tile_area,
                            min_mask_area,
                            min_mask_area_percent,
                            mask_filter_mode
                        )

                        if removed_components.sum() > 0:
                            # Create a mask with very small non-zero values for R and G channels
                            # This prevents numerical issues while keeping the color visually blue
                            removed_mask = removed_components.unsqueeze(-1).expand(-1, -1, 3)
                            removed_components_vis[0, y1:y2, x1:x2] = torch.where(
                                removed_mask > 0,
                                torch.tensor([1e-7, 1e-7, 1.0], dtype=torch.float32),
                                removed_components_vis[0, y1:y2, x1:x2]
                            )

                    # Dilate mask if requested
                    if dilate_masks > 0:
                        print(f"Dilating mask by {dilate_masks} pixels")
                        pre_dilate_sum = mask_result.sum()
                        mask_result = self.dilate_mask(mask_result, dilate_masks)
                        post_dilate_sum = mask_result.sum()
                        print(f"Mask sum before dilation: {pre_dilate_sum}, after dilation: {post_dilate_sum}")

                    # Update final mask
                    final_mask[0, y1:y2, x1:x2] = torch.logical_or(
                        final_mask[0, y1:y2, x1:x2],
                        mask_result if len(mask_result.shape) == 2 else mask_result[0]
                    ).float()

                    # Add colored mask to output
                    current_color = colors[bbox_idx]
                    mask_color_tensor = torch.tensor(current_color, dtype=torch.float32)
                    
                    # Expand mask for broadcasting
                    if len(mask_result.shape) == 3:
                        mask_for_color = mask_result[0]
                    else:
                        mask_for_color = mask_result
                    
                    # Apply color to mask
                    for c in range(3):
                        colored_masks[0, y1:y2, x1:x2, c] = torch.where(
                            mask_for_color > 0,
                            mask_for_color * mask_color_tensor[c] * mask_opacity + colored_masks[0, y1:y2, x1:x2, c] * (1 - mask_opacity),
                            colored_masks[0, y1:y2, x1:x2, c]
                        )

                    # Visualize bbox and tile
                    # Draw original bbox
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        linewidth=2,
                        edgecolor='red',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add object label
                    ax.text(bbox[0], bbox[1] - 5,
                           f' Object {bbox_idx} ',
                           color='red',
                           fontsize=8,
                           bbox=dict(
                               facecolor='white',
                               alpha=0.7,
                               edgecolor='none',
                               pad=0.3,
                               boxstyle='square'
                           ),
                           horizontalalignment='left',
                           verticalalignment='bottom')

                    # Draw tile
                    tile_rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor=distinct_colors[bbox_idx % len(distinct_colors)],
                        facecolor='none',
                        alpha=0.5
                    )
                    ax.add_patch(tile_rect)

            except Exception as e:
                print(f"Error processing bbox {bbox_idx}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue

        # Remove axes and padding
        ax.axis('off')
        ax.margins(0,0)
        ax.get_xaxis().set_major_locator(plt.NullLocator())
        ax.get_yaxis().set_major_locator(plt.NullLocator())
        
        # Save annotated image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        annotated_image_pil = Image.open(buf)
        plt.close(fig)

        # Convert annotated image to tensor
        annotated_image_tensor = torch.from_numpy(np.array(annotated_image_pil)[:, :, :3]).float() / 255.0
        annotated_image_tensor = annotated_image_tensor.unsqueeze(0)

        # Combine colored masks with original image
        final_colored_masks = torch.where(
            colored_masks > 0,
            colored_masks * mask_opacity + image * (1 - mask_opacity),
            image
        )

        # Create tensor for bounding boxes
        tile_bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

        # Combine removed components with original image using opacity
        final_removed_components = torch.where(
            removed_components_vis > 0,
            removed_components_vis * mask_opacity + image * (1 - mask_opacity),
            image
        )

        print("\nFinal results:")
        print(f"Final mask shape: {final_mask.shape}")
        print(f"Tile bboxes shape: {tile_bboxes_tensor.shape}")
        print(f"Annotated image shape: {annotated_image_tensor.shape}")
        print(f"Colored masks shape: {colored_masks.shape}")
        print(f"Removed components shape: {removed_components_vis.shape}")

        return (final_mask, tile_bboxes_tensor, annotated_image_tensor, final_removed_components, final_colored_masks)
     
NODE_CLASS_MAPPINGS = {
    "Sam2TiledSegmentation": Sam2TiledSegmentation,
    "Sam2ContextSegmentation": Sam2ContextSegmentation
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sam2TiledSegmentation": "Sam2TiledSegmentation",
    "Sam2ContextSegmentation": "Sam2ContextSegmentation"
}
