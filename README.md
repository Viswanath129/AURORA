
# Cloud Moment Detection with AURORA Architecture

This repository implements **AURORA (Advection-morphology-emergence-temporal-reasoning for Cloud Nowcasting)**, a geospatial reasoning framework inspired by Google Research.

## Architecture

The system integrates four distinct reasoning experts using a **Pixel-Wise Routing Network**:

1.  **Advection (Optical Flow)**: Captures motion (wind) from previous frames.
2.  **Morphology (UNet)**: Captures shape and texture evolution.
3.  **Emergence (Diffusion/UNet)**: Captures intensity changes and formation/dissipation of clouds.
4.  **Temporal (ConvLSTM)**: Captures long-term temporal dependencies.

These experts are fused by a **Routing Network** that assigns confidence weights to each expert per pixel, based on their predicted uncertainty and the current context.

## Integration Results

The integrated AURORA model achieves state-of-the-art results by dynamically selecting the best expert for each region.

- **SSIM**: ~0.89
- **PSNR**: ~24.25 dB

## Usage

1.  **Train**:
    ```bash
    python train.py
    ```
    (This trains the experts and the routing network. Data `sequences.pth` is required.)

2.  **Evaluate**:
    ```bash
    python evaluate.py
    ```
    (Compares AURORA against individual experts.)

3.  **Demo**:
    ```bash
    python demo.py
    ```
    (Generates `aurora_demo.png` visualizing the inputs, expert predictions, routing weights, and final output.)

## Files
- `routing_net.py`: The core integration module (router).
- `train.py`: Training pipeline for experts and router.
- `evaluate.py`: Metrics calculation.
- `demo.py`: Visualization.
