# AURORA: Cloud Moment Detection & Nowcasting
## Technical Architecture Report

### 1. Problem Statement
**"Cloud Moment Detection"** refers to the challenge of short-term weather forecasting (nowcasting, 0-2 hours) specifically focused on the rapid evolution of cloud cover.

Traditional Numerical Weather Prediction (NWP) models operate on physics equations that are computationally expensive and typically run with latencies of 3-6 hours. This latency makes them unsuitable for real-time applications such as:
*   **Solar Energy Grid Balancing:** Predicting abrupt cloud shading (ramp-down events).
*   **Aviation Safety:** Avoiding sudden turbulence or visibility loss.
*   **Meteorological Monitoring:** Tracking severe storm formation.

**The Core Challenge:** Standard deep learning models (like simple CNNs) typically produce blurry forecasts because they average out high-frequency details to minimize error (MSE). They fail to capture distinct physical phenomena like wind-driven motion (advection) vs. spontaneous cloud formation (emergence).

---

### 2. Our Approach: Mixture of Experts (MoE)
Instead of a single "black box" model, AURORA adopts a **"Divide and Conquer"** strategy. We assert that cloud dynamics are governed by distinct physical laws, and different neural architectures are better suited for each law.

We decompose the problem into four distinct reasoning paths (experts), which are then fused by a "Brain" (Routing Network) that dynamically decides which expert to trust for each pixel.

---

### 3. Architecture Breakdown

#### Path 1: Advection Expert (Optical Flow)
*   **Role:** Captures fluid motion (wind).
*   **Implementation:** Uses dense Optical Flow (e.g., Gunnar Farneback) to calculate motion vectors between previous frames and "warps" the current frame forward. This maintains high-frequency details for moving but stable clouds.

#### Path 2: Morphology Expert (UNet)
*   **Role:** Captures shape evolution and deformation.
*   **Implementation:** A classic UNet encoder-decoder architecture that learns local transformations (expansion, compression, shearing) that simple translation (optical flow) cannot model.

#### Path 3: Emergence Expert (Diffusion/Generative)
*   **Role:** Captures intensity changes, formation, and dissipation.
*   **Implementation:** A generative model (Diffusion-based or Generative UNet) trained to "hallucinate" plausible details where new clouds appear or existing ones fade, addressing the "blurriness" problem.

#### Path 4: Temporal Expert (ConvLSTM)
*   **Role:** Captures long-term dependencies and trends.
*   **Implementation:** A Convolutional LSTM (Long Short-Term Memory) network that maintains a memory state across the input sequence to understand acceleration or cyclic patterns.

#### The "Brain": Pixel-Wise Routing Network
*   **Role:** Uncertainty-aware fusion.
*   **Implementation:** A CNN taking all expert predictions and estimated *uncertainty maps* as input. It outputs a `Softmax` weight map (sum=1 per pixel).
    *   *Example:* If Optical Flow has high uncertainty in a region of new cloud formation, the router shifts weight to the Emergence Expert for those specific pixels.

---

### 4. Technology Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **Deep Learning** | PyTorch (Convolutional Layers, LSTMs, Autograd) |
| **Computer Vision** | OpenCV (Optical Flow), skimage (Metrics) |
| **Data Processing** | NumPy (Tensors), Matplotlib (Visualization) |
| **Frontend** | HTML5, CSS3 (Glassmorphism), JavaScript |
| **Patterns** | Strategy Pattern (Experts), Composite Pattern (Router) |

---

### 5. Programming & Implementation Details
The codebase is structured to enforce modularity:

*   **`train.py`**: Implements a multi-stage training pipeline.
    1.  Stage 1: Train experts independently.
    2.  Stage 2: Freeze experts, train the Routing Network.
*   **`routing_net.py`**: Contains the custom fusion logic using 1x1 convolutions to mix expert channels.
*   **`evaluate.py`**: Benchmarking script calculating SSIM (Structural Similarity) and PSNR (Peak Signal-to-Noise Ratio).
*   **Uncertainty Estimation**: Experts output 2 channels: `Prediction` (Mean) and `Uncertainty` (Variance), allowing the router to make informed decisions.


