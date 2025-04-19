This repository contains code related to the paper:  
**"CNN and Deep Operator Network Models to Predict Oil Production by Water Injection in 2D Pore Space Geometries."**

## Directory Structure

- `/code`: Contains all the code scripts and notebooks.
- `/BD_1`: Stores input/output data used for training the models.
- `/im`: Stores generated images.
- `/models`: Stores model checkpoints.

## Execution Workflow

The system is organized into three main stages. Execute the scripts in the following order:

### 1. Data Transformation
1. `[0]EfficientNetB7encoder.ipynb` — Transforms porous media images using EfficientNetB7.
2. `[0]X_PCA.ipynb` — Applies PCA to the transformed image representations.

### 2. Neural Network Training
3. `[1]predict_CNN_cv.py` — Trains CNN models using cross-validation.
4. `[1]predict_DON_cv.py` — Trains DeepONet models using cross-validation.

### 3. Metrics and Analysis
5. The following scripts generate the performance metrics presented in the paper:
   - `[2]histograms.py`
   - `[2]metrics_loss.py`
   - `[2]sorted_metrics_cnn.py`
   - `[2]sorted_metrics_don.py`
   - `[2]sorted_metrics_curves_1.py`
   - `[2]sorted_metrics_curves_2.py`
   - `[2]sorted_metrics_curves_3.py`

