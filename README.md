# SquatsModelTest

SquatsModelTest is a real-time squat classification demo built with MediaPipe Pose, OpenCV, and a TensorFlow/Keras model. It uses webcam input to detect body landmarks, normalize the pose, and classify squat form live.

## What the model does

The current pipeline in `live2.py` performs these steps:

1. Detect pose landmarks from the webcam with MediaPipe.
2. Keep 12 key landmarks for the lower and upper body.
3. Normalize the skeleton so the model is less sensitive to camera position and scale.
4. Convert each frame into 10 biomechanical angle features.
5. Build a 45-frame sequence window.
6. Run the sequence through a Keras classifier and smooth the output with an exponential moving average.
7. Display one of these classes:
   - Incorrect Posture
   - Legs too Narrow
   - Legs too Wide
   - Not a Squat
   - Perfect Squats

The squat rep counter is driven by knee-angle thresholds, not by the classifier output.

## Deep Learning architecture

The project uses a sequence-based squat classifier built on top of pose landmarks from MediaPipe.

### Input pipeline

1. MediaPipe Pose detects body landmarks from the webcam.
2. `live2.py` keeps 12 landmarks focused on the torso, hips, knees, ankles, heels, and feet.
3. The skeleton is normalized by centering on the hips, rotating to reduce camera angle effects, and scaling by body size.
4. Each frame is converted into 10 angle-based features.
5. A sliding window of 45 frames is passed to the model.

### Model structure

The training notebook in `Colab/squat_classifier_final_1 (4).ipynb` contains three architecture variants:

1. `1D CNN -> BiLSTM -> Multi-Head Self-Attention -> Dense classifier` with raw coordinates + 8 angle features.
2. A deeper `1D CNN -> BiLSTM -> Multi-Head Self-Attention -> Dense classifier` variant with 10 angle features and a larger dense head.
3. `TCN -> Feature Fusion -> Dense classifier` for an angle-only setup, using temporal convolution blocks and a lightweight classification head.

The notebook shows the evolution of the model, and the final section is the compact angle-only version.

In simple terms:

- `1D CNN` learns short-term motion patterns from the pose sequence.
- `BiLSTM` captures movement over time in both forward and backward directions.
- `Multi-Head Self-Attention` helps the model focus on the most important frames in the squat sequence.
- `Dense classifier` outputs probabilities for the 5 squat classes.

### Live inference

The live demo in `live2.py` uses the exported Keras model from `Squats_Model/` and applies:

- feature scaling with `scaler_params (9).json`
- EMA smoothing for more stable predictions
- a confidence threshold before showing the final label

Supporting files:

- `Squats_Model/best_model (9).keras` - trained Keras model
- `Squats_Model/scaler_params (9).json` - feature normalization values

There is also an older variant in `live.py` that uses a different feature format and a different model/scaler pair.

## Setup

1. Create and activate a Python virtual environment.
2. Install the project dependencies:

```bash
pip install opencv-python mediapipe numpy tensorflow
```

3. Make sure the model paths inside `live2.py` point to the files in `Squats_Model/`.
4. Connect a webcam.

## Run

```bash
python live2.py
```

Controls:

- `Q` - quit
- `R` - reset rep counter

## Notes

- `live2.py` is the current angle-only version and is the best starting point.
- `live.py` is an older feature-rich version kept for comparison.
- The `All Models/` and `Colab/` folders contain training artifacts and notebook history.