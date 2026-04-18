# Article and Benchmarking Guide

This document provides usage instructions for all scripts related to article preparation and model benchmarking.

## Table of Contents

1. [File Structure](#file-structure)
2. [Training Scripts](#training-scripts)
3. [Benchmarking Scripts](#benchmarking-scripts)
4. [Visualization Scripts](#visualization-scripts)
5. [Article Files](#article-files)
6. [Quick Start Guide](#quick-start-guide)

---

## File Structure

### Training Scripts
- `train.py` - Original multi-task training (your original file, unchanged)
- `trainer.py` - Original trainer class (your original file, unchanged)
- `train_depth_baseline.py` - Single-task depth-only training for KITTI benchmarking
- `train_seg_baseline_coco.py` - Single-task segmentation training on COCO (template)

### Benchmarking Scripts
- `benchmark_mtl.py` - Your original multi-task benchmark
- `benchmark_depth_kitti.py` - Standard KITTI Eigen split evaluation
- `benchmark_seg_coco.py` - COCO segmentation evaluation (template)
- `benchmark_seg_kitti_cityscapes.py` - Cross-dataset evaluation

### Visualization Scripts (for Article Figures)
- `visualize_seg_masks.py` - **Main script** for generating segmentation masks
- `visualize_comparison.py` - Creates publication-ready comparison figures
- `generate_qualitative_results.py` - Batch processing for multiple images
- `test_mtl.py` - Your original testing script with bounding boxes

### Article Files
- `article/mine/main.tex` - Original article template
- `article/mine/main_v2.tex` - Article with benchmarking methodology
- `article/mine/main_v3.tex` - **Recommended** - Article with qualitative evaluation focus
- `article/mine/references.bib` - BibTeX references

---

## Training Scripts

### 1. Train Multi-Task Model (Original)

Your existing training pipeline:

```bash
python train.py \
    --model_name mtl_model \
    --data_path ./kitti_data \
    --split eigen_zhou \
    --num_epochs 50 \
    --batch_size 16
```

**Trains:** Depth + Segmentation + Centers jointly

### 2. Train Depth-Only Baseline

For fair comparison with Monodepth2 and Lite-Mono:

```bash
python train_depth_baseline.py \
    --model_name depth_baseline \
    --data_path ./kitti_data \
    --split eigen_zhou \
    --model lite-mono-8m \
    --height 192 \
    --width 640 \
    --batch_size 12 \
    --num_epochs 50
```

**Outputs:**
- Single-task depth model
- Compatible with standard KITTI evaluation protocol

### 3. Train Segmentation on COCO (Template)

For quantitative segmentation benchmarking:

```bash
python train_seg_baseline_coco.py \
    --model_name seg_coco_baseline \
    --data_path ./coco \
    --model lite-mono-8m \
    --num_classes 133 \
    --height 640 \
    --width 640 \
    --num_epochs 100
```

**Note:** Requires COCO dataset implementation (template provided)

---

## Benchmarking Scripts

### 1. Depth Evaluation on KITTI

Standard evaluation following Monodepth2/Lite-Mono protocol:

```bash
python benchmark_depth_kitti.py \
    --load_weights_folder ./tmp/depth_baseline/models/weights_49 \
    --model lite-mono-8m \
    --data_path ./kitti_raw \
    --height 192 \
    --width 640 \
    --eval_split eigen
```

**Outputs:**
```
Results for LITE-MONO-8M on KITTI eigen Split
==================================================
    Metric |      value |      value |      value
--------------------------------------------------
   abs_rel: |        -- |        -- |        --
   sq_rel: |        -- |        -- |        --
    rmse: |        -- |        -- |        --
rmse_log: |        -- |        -- |        --
      a1: |        -- |        -- |        --
==================================================
Scaling ratios - med: X.XXX, std: X.XXX
```

**Metrics Explained:**
- `abs_rel`: Absolute Relative Error (lower is better)
- `sq_rel`: Squared Relative Error (lower is better)
- `rmse`: Root Mean Squared Error (lower is better)
- `rmse_log`: RMSE in log space (lower is better)
- `a1`, `a2`, `a3`: Threshold accuracy (higher is better)

### 2. Compare MTL vs Single-Task Depth

Evaluate your multi-task model's depth head:

```bash
python benchmark_depth_kitti.py \
    --load_weights_folder ./tmp/mtl_model/models/weights_49 \
    --model lite-mono-8m
```

Compare with the single-task baseline to see if multi-task learning helps or hurts depth performance.

### 3. Segmentation Evaluation (Qualitative)

Since you don't have COCO/Cityscapes ground truth:

```bash
python benchmark_seg_kitti_cityscapes.py \
    --load_weights_folder ./tmp/mtl_model/models/weights_49 \
    --kitti_data_path ./kitti_raw
```

This explains the limitation and suggests alternatives.

---

## Visualization Scripts

### 1. Main Visualization Script

**`visualize_seg_masks.py`** - Generates full segmentation masks for article figures

```bash
python visualize_seg_masks.py \
    --image_path ./kitti_data/2011_09_26/image_02/data/0000000000.png \
    --load_weights_folder ./tmp/mtl_model/models/weights_49 \
    --model lite-mono-8m \
    --output_dir ./article_figures \
    --save_overlay \
    --save_individual_classes \
    --alpha 0.5
```

**Arguments:**
- `--image_path`: Path to image or directory of images
- `--load_weights_folder`: Path to trained model weights
- `--model`: Model variant (lite-mono, lite-mono-8m, etc.)
- `--output_dir`: Where to save outputs (default: same as input)
- `--save_overlay`: Include overlay of mask on original image
- `--save_individual_classes`: Save separate mask for each class
- `--alpha`: Transparency for overlay (0.0-1.0, default 0.5)

**Generated Files per Image:**
```
{name}_semantic_mask.png      - Full colored semantic mask (MAIN OUTPUT)
{name}_overlay.png            - Mask overlay on original image
{name}_depth_viz.png          - Depth map visualization (plasma colormap)
{name}_centers_viz.png        - Instance center heatmap
{name}_combined.png           - 6-panel combined figure
{name}_class_masks/           - Directory with individual class masks
    ├── {name}_Person.png
    ├── {name}_Car.png
    ├── {name}_Bicycle.png
    └── ... (one per class present)
```

**Class Colors (BGR format for OpenCV):**
| Class | Color |
|-------|-------|
| Background | Black (0,0,0) |
| Person | Red (255,0,0) |
| Bicycle | Green (0,255,0) |
| Car | Blue (0,0,255) |
| Motorcycle | Cyan (255,255,0) |
| Bus | Magenta (255,0,255) |
| Truck | Yellow (0,255,255) |
| Traffic Light | Light Green (128,255,0) |
| Stop Sign | Orange (255,128,0) |

### 2. Publication-Ready Comparison Figure

**`visualize_comparison.py`** - Creates 5-panel paper figure

```bash
python visualize_comparison.py \
    --image_path ./kitti_data/sample.png \
    --load_weights_folder ./tmp/mtl_model/models/weights_49 \
    --model lite-mono-8m \
    --output_path ./article_figures/paper_fig_1.png \
    --dpi 300
```

**Figure Layout:**
```
+--------+--------+--------+--------+--------+
|        |        |        | Overlay|        |
| Input  | Depth  |Semantic|   +    | Legend |
| Image  |  Map   |  Mask  | Centers|        |
|  (a)   |  (b)   |  (c)   | (d,e)  |        |
+--------+--------+--------+--------+--------+
```

**High DPI Options:**
- `--dpi 150` - Good for web/preview
- `--dpi 300` - Publication quality (recommended)
- `--dpi 600` - High-end publication (large file)

### 3. Batch Generate Multiple Results

**`generate_qualitative_results.py`** - Process multiple images

```bash
python generate_qualitative_results.py \
    --image_dir ./kitti_data/2011_09_26/image_02/data/ \
    --load_weights_folder ./tmp/mtl_model/models/weights_49 \
    --output_dir ./article_figures/batch_results \
    --num_images 20 \
    --seed 42 \
    --model lite-mono-8m
```

**Arguments:**
- `--image_dir`: Directory containing images
- `--num_images`: Number of images to process (0 = all)
- `--seed`: Random seed for reproducible selection

**Output Structure:**
```
article_figures/batch_results/
├── image_001_semantic_mask.png
├── image_001_overlay.png
├── image_001_depth_viz.png
├── image_001_centers_viz.png
├── image_001_combined.png
├── image_001_paper_fig.png
├── image_001_class_masks/
│   ├── image_001_Car.png
│   ├── image_001_Person.png
│   └── ...
├── image_002_semantic_mask.png
└── ...
```

---

## Article Files

### Recommended Version: `main_v3.tex`

This version includes:
- Qualitative evaluation methodology
- Clear explanation of pseudo-labels limitation
- Figures for semantic masks and overlays
- Honest discussion of future work

### Compiling the Article

```bash
cd article/mine

# Compile once
pdflatex main_v3.tex

# Handle bibliography
bibtex main_v3

# Compile twice more for references
pdflatex main_v3.tex
pdflatex main_v3.tex
```

**Alternative (if using modern LaTeX):**
```bash
cd article/mine
latexmk -pdf main_v3.tex
```

### Adding Figures to Article

Example LaTeX code for including segmentation masks:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\columnwidth]{figures/sample_semantic_mask.png}
\caption{Semantic segmentation mask showing per-pixel class predictions. 
         Colors: red=person, blue=car, green=bicycle, etc.}
\label{fig:semantic_mask}
\end{figure}
```

For side-by-side comparison:

```latex
\begin{figure*}[htbp]
\centering
\begin{subfigure}[b]{0.24\textwidth}
    \includegraphics[width=\textwidth]{figures/input.png}
    \caption{Input}
\end{subfigure}
\begin{subfigure}[b]{0.24\textwidth}
    \includegraphics[width=\textwidth]{figures/depth.png}
    \caption{Depth}
\end{subfigure}
\begin{subfigure}[b]{0.24\textwidth}
    \includegraphics[width=\textwidth]{figures/semantic_mask.png}
    \caption{Semantic Mask}
\end{subfigure}
\begin{subfigure}[b]{0.24\textwidth}
    \includegraphics[width=\textwidth]{figures/overlay.png}
    \caption{Overlay}
\end{subfigure}
\caption{Multi-task outputs}
\end{figure*}
```

---

## Quick Start Guide

### Step 1: Train Baseline Models

```bash
# Train depth-only baseline for benchmarking
python train_depth_baseline.py \
    --model_name depth_baseline \
    --num_epochs 50

# Train multi-task model
python train.py \
    --model_name mtl_model \
    --num_epochs 50
```

### Step 2: Benchmark Depth

```bash
# Evaluate depth-only baseline
python benchmark_depth_kitti.py \
    --load_weights_folder ./tmp/depth_baseline/models/weights_49

# Evaluate MTL depth head
python benchmark_depth_kitti.py \
    --load_weights_folder ./tmp/mtl_model/models/weights_49
```

### Step 3: Generate Qualitative Results

```bash
# Generate segmentation masks for article figures
python visualize_seg_masks.py \
    --image_path ./sample_images/ \
    --load_weights_folder ./tmp/mtl_model/models/weights_49 \
    --output_dir ./article/mine/figures/results
```

### Step 4: Create Publication Figures

```bash
# Create high-quality comparison figure
python visualize_comparison.py \
    --image_path ./sample_images/best_result.png \
    --load_weights_folder ./tmp/mtl_model/models/weights_49 \
    --output_path ./article/mine/figures/figure_1.png \
    --dpi 300
```

### Step 5: Compile Article

```bash
cd article/mine
pdflatex main_v3.tex
bibtex main_v3
pdflatex main_v3.tex
pdflatex main_v3.tex
```

---

## Tips for Publication

### Selecting Images for Figures

Choose images that show:
1. **Multiple object classes** (car + person + bicycle)
2. **Varying scales** (close cars and distant cars)
3. **Occlusions** (cars partially hidden)
4. **Complex scenes** (intersections, traffic)
5. **Good lighting** (avoid overexposure/shadows)

### Figure Quality Checklist

- [ ] Resolution: At least 300 DPI
- [ ] Font size: Legible in print
- [ ] Colorblind-friendly: Avoid red-green combinations
- [ ] Class legend: Clear color-to-class mapping
- [ ] Consistent: Same color scheme across all figures

### Common Issues

**Issue:** Segmentation masks look noisy
**Solution:** Increase training epochs or adjust class weights in trainer.py

**Issue:** Depth predictions are blurry
**Solution:** This is normal for self-supervised methods; ensure you're using median scaling during evaluation

**Issue:** Small objects not detected
**Solution:** Visualize the `centers_viz.png` - if peaks are too weak, lower the threshold in panoptic post-processing

---

## Summary of Key Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `train_depth_baseline.py` | Train single-task depth | Benchmarking vs Monodepth2 |
| `benchmark_depth_kitti.py` | Evaluate depth on KITTI | Get standard metrics |
| `visualize_seg_masks.py` | Generate segmentation masks | **Main script for article** |
| `visualize_comparison.py` | Create paper figures | Publication-ready images |
| `generate_qualitative_results.py` | Batch process | Multiple examples for article |

---

## References

### Key Papers for Comparison

**Depth:**
- Monodepth2 (Godard et al., ICCV 2019) - Abs Rel: 0.115
- Lite-Mono (Zhang et al., CVPR 2023) - Abs Rel: 0.097 (Lite-Mono-8M)

**Segmentation:**
- Panoptic-DeepLab (Cheng et al., CVPR 2020) - PQ: 46.5% (COCO)
- Mask2Former (Cheng et al., CVPR 2022) - PQ: 57.0% (Swin-L)

**Multi-Task:**
- OmniDet (Ramesh et al., RA-L 2021) - 6 tasks
- HydraNet (Muller et al., IJCV 2018) - Efficient MTL

---

## Contact & Support

For questions about:
- **Training issues**: Check GPU memory, reduce batch size if needed
- **Evaluation**: Ensure you're using the correct KITTI split (eigen_zhou)
- **Visualization**: Verify model weights loaded correctly

---

*Last updated: April 2026*
