# 🐱🐶 Cat vs Dog Image Classification
A Deep Learning Project using CNNs (Keras/TensorFlow)
This repository contains a complete end-to-end implementation of a Cat vs Dog Image Classification model using a Convolutional Neural Network (CNN).
The project includes dataset loading, preprocessing, model building, training, evaluation, and visualization of results.

# 📌 Project Overview
This project classifies images into two categories:\
0 → Cat\
1 → Dog\
A custom CNN architecture is used for training, built using TensorFlow Keras.\
The dataset consists of image folders:\
/animals/cat\
/animals/dog\
Images are loaded from disk, resized to 128×128, normalized, and fed into the model.

# 🗂 Project Structure
📁 Cat-Dog-Classification\
│── cat_dog.ipynb        # Main notebook with full implementation\
|__ data.txt             # Link to the dataset\
│── README.md            # Project documentation


# ⚙️ How It Works
# 1️⃣ Loading Libraries
The project uses:\
TensorFlow/Keras\
NumPy\
PIL (Pillow)\
Matplotlib\
Scikit-learn

# 2️⃣ Loading Images
Images are read from folder paths using a custom function:\
def load_images(folder, label):\
    # Loads images and assigns labels (0 = cat, 1 = dog)\
It also handles corrupted image files safely using try-except.

# 3️⃣ Dataset Preparation
Convert list of images → NumPy arrays\
Normalize pixel values\
One-hot encode labels\
Train–Test split (80%-20%)

# 4️⃣ Model Architecture
A simple CNN with:\
Conv2D(32) → MaxPool\
Conv2D(64) → MaxPool\
Flatten\
Dense(64)\
Dense(2, softmax)\
Activation functions: ReLU and Softmax\
Loss: Categorical Crossentropy\
Optimizer: Adam

# 5️⃣ Training
model.fit(\
    X_train, y_train,\
    validation_split=0.2,\
    steps_per_epoch=2,\
    epochs=2,\
    batch_size=8\
)

# 6️⃣ Evaluation
The model is evaluated on unseen test data:\
model.evaluate(X_test, y_test)

# 7️⃣ Visualization
Accuracy vs Epochs\
Loss vs Epochs\
Both are plotted using Matplotlib.
