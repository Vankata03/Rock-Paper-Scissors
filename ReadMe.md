# ✊✌️🖐️ AI Rock Paper Scissors

> A real-time hand gesture recognition game built with Python, OpenCV, and MediaPipe. Play Rock-Paper-Scissors against an AI that sees your moves!

## 📜 About the Project

This project uses computer vision and machine learning to recognize hand gestures in real-time. It detects the landmarks on your hand, classifies the gesture as **Rock**, **Paper**, or **Scissors**, and lets you play a classic game against the computer.

It includes a complete pipeline: from collecting your own training data to training a custom model and playing the game with a polished UI.

## ✨ Features

- **Real-time Detection**: Uses MediaPipe Hands for fast and accurate hand tracking.
- **Custom AI Model**: Train a Random Forest classifier on your own hand gestures.
- **Interactive Gameplay**:
  - **Basic Mode (`play.py`)**: Simple text-based feedback.
  - **Pro Mode (`play2.py`)**: Enhanced UI with Pillow, color-coded results (Green for Win, Red for Loss), and smooth overlays.
- **Data Collection Tool**: Easily build a dataset by recording gestures with a single key press.

## 🛠️ Technologies Used

*   **Python 3.x**
*   **OpenCV** (Computer Vision)
*   **MediaPipe** (Hand Tracking)
*   **Scikit-learn** (Machine Learning)
*   **Pandas & NumPy** (Data Processing)
*   **Pillow (PIL)** (UI/Text rendering)

## 📦 Installation

1.  **Clone the repository**
    git clone https://github.com/yourusername/hand-gesture-game.git
    cd hand-gesture-game
    2.  **Install dependencies**
    You can install the required libraries using pip:
    pip install opencv-python mediapipe scikit-learn pandas numpy pillow
    ## 🚀 How to Run

Follow these three steps to get the game running perfectly on your machine.

### Step 1: Collect Data (Optional if model exists)
If you want to train the model on *your* specific hand movements:
1.  Run the data collection script:
    python collect_data.py
    2.  **Controls**:
    - Press **`0`** to record **Rock** samples.
    - Press **`1`** to record **Scissors** samples.
    - Press **`2`** to record **Paper** samples.
    - Press **`Q`** to quit.
3.  Try to collect at least 100-200 samples per gesture for better accuracy.

### Step 2: Train the Model
Once `hand_data.csv` is generated (or if you use the existing one), train the AI:
python train.py*   This will create a `model.p` file containing the trained Random Forest classifier.
*   It usually takes just a few seconds and will output the model's accuracy (e.g., "Accuracy: 99.5%").

### Step 3: Play the Game!
Now you are ready to play. You have two versions:

**Option A: Enhanced UI Version (Recommended)**
python play2.py*   **Spacebar**: Lock in your move and play against the Computer.
*   **Q**: Quit the game.

**Option B: Basic Version**
python play.py## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `collect_data.py` | Captures webcam video, extracts hand landmarks, and saves them to `hand_data.csv`. |
| `train.py` | Loads the CSV data, trains a Random Forest Classifier, and saves it as `model.p`. |
| `play2.py` | The main game file with a polished UI, score logic, and visual feedback. |
| `play.py` | A simpler version of the game logic. |
| `model.p` | The serialized trained machine learning model. |
| `hand_data.csv` | The dataset containing landmark coordinates for each gesture. |

## 🎮 Controls Summary

| Key | Action | Context |
| :--- | :--- | :--- |
| **Space** | Play Hand / Reveal Winner | Game Mode |
| **0** | Record Rock | Collection Mode |
| **1** | Record Scissors | Collection Mode |
| **2** | Record Paper | Collection Mode |
| **Q** | Quit Application | Anywhere |

---
*Created with ❤️ by Vankata*