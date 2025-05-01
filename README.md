# Evaluating the Efficiency of Context-Based Segmentation

## Introduction

Automatic image segmentation has become a vital tool in various computer vision tasks, with models like SAM2 offering significant advancements. However, despite the performance of large models, SAM2 sometimes struggles with small or intricate objects, even when precise bounding boxes are provided. The integration of context-aware models seeks to address this limitation by improving segmentation accuracy, particularly for smaller objects. This study explores whether incorporating contextual information into the segmentation process provides measurable improvements or if alternative approaches, such as using the **Base+** model, can achieve similar results with greater efficiency.

## Problem Statement

While SAM2's large model performs well on general objects, it exhibits a tendency to misinterpret small objects or fine details, leading to inaccuracies in segmentation, especially when smaller objects appear in complex scenes. On the other hand, smaller SAM2 models tend to perform better in fine-grained tasks but fail to handle large objects adequately. To address these challenges, a context-based approach has been proposed, where the input image is cropped around the bounding box of the target object, providing the model with a more focused view.

## Methodology

In this study, we tested the segmentation performance using various approaches:

1. **Context-Based Segmentation**: This approach crops the input image around each bounding box, ensuring that the SAM2 model processes only the target object, potentially improving accuracy for small objects.
2. **Tiled Segmentation**: A tiling-based approach that slices the image into smaller sections to avoid performance degradation but without focusing on the context of specific objects.
3. **Large SAM2.1**: The base large model, known for its higher capacity, but prone to errors when small objects are present.
4. **Base+ SAM2.1**: An alternative model that balances computational efficiency and segmentation quality by offering a more streamlined approach, often without needing extensive context processing.
5. **Small SAM2.1**: A version of the SAM2 model with fewer parameters, optimized for faster processing and lower computational demands. While it performs better on small objects and fine details, it may struggle with larger or more complex scenes due to its reduced capacity compared to the large model.

The testing was conducted using a set of objects such as golf balls, golf clubs, human faces, and various combinations of items (e.g., shoes, hands, pants).

## Results

| Prompt        | Original Image   | Florence Bboxes  | Context Node     | Tiled Node       | Large SAM2.1     | Base+ SAM2.1     | Small SAM2.1     |
|---------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| golf ball     | ![][gb_orig]     | ![][gb_flor]     | ![][gb_ctx]      | ![][gb_tiled]    | ![][gb_large]    | ![][gb_base+]    | ![][gb_small]    |
| golf club     | ![][gc_orig]     | ![][gc_flor]     | ![][gc_ctx]      | ![][gc_tiled]    | ![][gc_large]    | ![][gc_base+]    | ![][gc_small]    |
| human face    | ![][hf_orig]     | ![][hf_flor]     | ![][hf_ctx]      | ![][hf_tiled]    | ![][hf_large]    | ![][hf_base+]    | ![][hf_small]    |
| shoes, hands, cap, mouth, pants, ball, watch  | ![][mix_orig]    | ![][mix_flor]    | ![][mix_ctx]     | ![][mix_tiled]   | ![][mix_large]   | ![][mix_base+]   | ![][mix_small]   |

- **Contextual and Tiled Nodes**: Both utilized the SAM2.1 large model for segmentation.
- **Node Settings**: Identical across all tests to ensure a fair comparison.

<!-- Image aliases -->

[gb_orig]: test_images/comparisons/1_original%20image.jpg  
[gb_flor]: test_images/comparisons/1_golf%20ball_florence.jpg  
[gb_ctx]: test_images/comparisons/1_golf%20ball_context.jpg  
[gb_tiled]: test_images/comparisons/1_golf%20ball_tiled.jpg  
[gb_large]: test_images/comparisons/1_golf%20ball_large.jpg  
[gb_base+]: test_images/comparisons/1_golf%20ball_baseplus.jpg  
[gb_small]: test_images/comparisons/1_golf%20ball_small.jpg  

[gc_orig]: test_images/comparisons/1_original%20image.jpg  
[gc_flor]: test_images/comparisons/1_golf%20club_florence.jpg  
[gc_ctx]: test_images/comparisons/1_golf%20club_context.jpg  
[gc_tiled]: test_images/comparisons/1_golf%20club_tiled.jpg  
[gc_large]: test_images/comparisons/1_golf%20club_large.jpg  
[gc_base+]: test_images/comparisons/1_golf%20club_baseplus.jpg  
[gc_small]: test_images/comparisons/1_golf%20club_small.jpg  

[hf_orig]: test_images/comparisons/1_original%20image.jpg  
[hf_flor]: test_images/comparisons/1_human%20face_florence.jpg  
[hf_ctx]: test_images/comparisons/1_human%20face_context.jpg  
[hf_tiled]: test_images/comparisons/1_human%20face_tiled.jpg  
[hf_large]: test_images/comparisons/1_human%20face_large.jpg  
[hf_base+]: test_images/comparisons/1_human%20face_baseplus.jpg  
[hf_small]: test_images/comparisons/1_human%20face_small.jpg  

[mix_orig]: test_images/comparisons/2_original%20image.jpg  
[mix_flor]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20watch_florence.jpg  
[mix_ctx]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20watch_context.jpg  
[mix_tiled]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20watch_tiled.jpg  
[mix_large]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20watch_large.jpg  
[mix_base+]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20watch_baseplus.jpg  
[mix_small]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20watch_small.jpg

## Analysis

The results indicate that **Base+** consistently performed on par with or exceeded the accuracy of the context-based approach. In several test cases, **Base+** even outperformed the context-based model, suggesting that the added computational complexity of context processing might not always yield substantial improvements. Additionally, **Base+** provided a more efficient segmentation without sacrificing detail, especially in scenarios where context processing was not feasible or necessary.

- **Context-Based Segmentation**: This approach demonstrated noticeable improvements in handling small objects by focusing exclusively on the cropped areas around the bounding boxes. However, this increased segmentation accuracy came at the cost of additional computation and image manipulation.
- **Base+**: Offered an efficient alternative that maintained a high level of accuracy while avoiding the complexity of cropping and reprocessing images. In many cases, **Base+** provided results comparable to context-based segmentation without the need for additional context-aware image preparation.

## Conclusion

The evaluation suggests that while context-based segmentation models can enhance accuracy, particularly for small objects, **Base+** offers a comparable or even superior alternative in many scenarios. It provides a promising approach for users who seek a balance between segmentation quality and computational efficiency. 

For researchers or developers interested in experimenting with these models, the context-based approach can still offer benefits in specific cases where detailed object isolation is crucial. However, in most practical applications, **Base+** may be the preferred model for achieving high-quality segmentation without the added complexity of context processing.

---

## Available Nodes

| Node                     | Type              | Description                                                                                       | Image                                                                 |
|--------------------------|-------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `Sam2ContextSegmentation` | ✅ Main Node       | Segments objects from bounding boxes by generating crops centered on each box. Uses local context to guide SAM2 toward cleaner and more coherent results. | ![Sam2ContextSegmentation](test_images/nodes/Sam2ContextSegmentation.png) |
| `Sam2TiledSegmentation`   | ⚠️ Alternative     | Segmentation via regular tiling using [SAHI](https://github.com/obss/sahi). Works in simple cases but is less accurate than the contextual approach. | ![Sam2TiledSegmentation](test_images/nodes/Sam2TiledSegmentation.png)     |

---

## Context Process

| Output                    | Description                                                                                          | Image                  |
|---------------------------|------------------------------------------------------------------------------------------------------|------------------------|
| Original Image            | Input image containing one or more objects to be segmented                                           | ![][orig_img]          |
| Florence Bounding Boxes   | Image annotated with bounding boxes predicted by Florence 2 for each detected object (ball, glove)   | ![][florence_bboxes]   |
| Context Tiles             | Contextual crops centered on each bounding box, used as input for SAM2                               | ![][context_tiles]     |
| Colored Masks             | Per-object segmentation masks overlaid with unique colors for visual clarity                         | ![][colored_masks]     |
| Cleaned Mask              | Composite mask with small, disconnected regions removed to reduce noise (if setting is enabled)      | ![][cleaned_mask]      |
| Final Mask                | Final mask combining all cleaned segments, used for downstream processing                            | ![][final_mask]        |

<!-- Image aliases -->

[orig_img]: test_images/context_process/original%20image.jpg  
[florence_bboxes]: test_images/context_process/florence%20bounding%20boxes.jpg  
[context_tiles]: test_images/context_process/context%20tiles.jpg  
[colored_masks]: test_images/context_process/colored%20masks.jpg  
[cleaned_mask]: test_images/context_process/cleaned%20mask.jpg  
[final_mask]: test_images/context_process/final%20mask.jpg  

---

## Installation

To install these custom nodes, clone or download this repository into your `ComfyUI/custom_nodes/` folder.

After restarting ComfyUI, the nodes `Sam2ContextSegmentation` and `Sam2TiledSegmentation` will appear and be ready to use.

A complete example workflow is included in the `workflows` folder to demonstrate how to use the context-based segmentation effectively.

---

## Credits

Thanks to the following open-source projects for their valuable contributions:

- [Florence 2](https://www.microsoft.com/en-us/research/publication/florence-2-advancing-a-unified-representation-for-a-variety-of-vision-tasks/)
- [Segment Anything 2](https://github.com/facebookresearch/segment-anything)
- [SAHI](https://github.com/obss/sahi)
- [ComfyUI Segment Anything 2 (Kijai)](https://github.com/kijai/ComfyUI-segment-anything-2)

Images include content sourced from [federgolf.it](https://www.federgolf.it) and a frame from a UEFA Champions League 2010 broadcast.
