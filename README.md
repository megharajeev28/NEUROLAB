# 🧠 NEURO LAB — Interactive Neural Network Learning Laboratory

**An interactive educational application for visualizing and understanding neural network training using Scilab.**

Developed for the **Scilab GUIVerse Hackathon 2026**, organized by the FOSSEE Project, IIT Bombay.

![Scilab](https://img.shields.io/badge/Platform-Scilab-blue)
![Domain](https://img.shields.io/badge/Domain-AI%20%2F%20Machine%20Learning-purple)
![Version](https://img.shields.io/badge/Scilab-2026.1.0-orange)
![Project](https://img.shields.io/badge/Type-Educational%20Tool-green)

---

## 📌 Overview

Neural networks are a fundamental component of Artificial Intelligence and Machine Learning. However, beginners often find it difficult to understand how data flows through neurons, how weights are updated, and how a network learns from its errors.

Traditional learning approaches often rely on mathematical equations, static diagrams, and theoretical explanations, making the training process difficult to visualize.

**NEURO LAB** addresses this challenge by providing an interactive graphical environment where users can train a small neural network, experiment with different parameters, and observe how these changes affect predictions, accuracy, loss, and network structure.

Built using **Scilab 2026.1.0**, NEURO LAB transforms abstract neural network concepts into a hands-on learning experience.

## 🎯 Objectives

- Demonstrate the fundamental working of an Artificial Neural Network (ANN).
- Visualize forward propagation, backpropagation, and gradient descent.
- Allow users to experiment with neural network training parameters.
- Display training loss, prediction probabilities, and classification accuracy.
- Visualize the network architecture and learned weight signs.
- Connect mathematical concepts with observable training results.

## ✨ Features

### 1. Interactive Dataset Selection

Experiment with three classic logical datasets:

- **XOR:** Learn a non-linearly separable logical relationship.
- **AND:** Predict whether both inputs are 1.
- **OR:** Predict whether at least one input is 1.

Each dataset contains four possible combinations of two binary inputs.

### 2. Customizable Training Parameters

Adjust the following parameters directly through the GUI:

| Parameter | Description |
|---|---|
| Learning Rate | Controls the size of weight updates during training. |
| Epochs | Specifies the number of training iterations. |
| Hidden Neurons | Determines the number of neurons in the hidden layer. |

Experiment with different settings to observe how they influence learning.

### 3. Neural Network Visualization

Visualize the network's architecture:

**Input Layer → Hidden Layer → Output Layer**

- Two input neurons representing `X1` and `X2`.
- A configurable hidden layer.
- One output neuron for binary classification.
- Connections representing learned weights.
- Color-coded weight signs: cyan for positive weights and red for negative weights.

The visualization is generated from the trained network's weight matrices.

### 4. Training Loss Graph

Observe how the loss changes over successive training epochs.

- **X-axis:** Epoch
- **Y-axis:** Training loss

A decreasing loss generally indicates that the model is improving its predictions on the training data.

### 5. Prediction Table

Inspect the network's predictions for all four input combinations.

| Column | Description |
|---|---|
| X1, X2 | Input values |
| Actual | Expected output |
| Predicted | Predicted binary class |
| Probability | Output probability before thresholding |

Probabilities greater than or equal to `0.5` are classified as class `1`; otherwise, the prediction is class `0`.

### 6. Accuracy and Loss Metrics

After training, the application displays classification accuracy and final loss.

Accuracy is calculated as:

`Accuracy = (Correct Predictions / Total Predictions) × 100`

### 7. Network Inspection

Use the **INSPECT NETWORK** button to examine:

- Number of input, hidden, and output neurons.
- Weight matrix dimensions.
- A sample prediction probability.

### 8. Mathematics Explorer

The **SHOW MATHEMATICS** feature explains the mathematical operations behind the network, helping users connect the visual interface with the underlying algorithms.

### 9. Reset Functionality

Reset the application to its initial state, including the selected dataset, training parameters, displayed results, graphs, and network visualization.

## 🏗️ Neural Network Architecture

NEURO LAB implements a feed-forward neural network with one hidden layer.

```text
         INPUT LAYER
          X1    X2
           \    /
            \  /
         HIDDEN LAYER
       H1  H2  H3 ... Hn
             |
             |
         OUTPUT LAYER
              Y
```

The hidden layer size can be adjusted through the GUI.

### Mathematical Formulation

**Hidden layer**

```text
Z1 = X · W1 + b1
A1 = sigmoid(Z1)
```

**Output layer**

```text
Z2 = A1 · W2 + b2
A2 = sigmoid(Z2)
```

Where:

- `X` represents the input data.
- `W1` and `W2` represent the weight matrices.
- `b1` and `b2` represent bias terms.
- `A1` represents hidden-layer activations.
- `A2` represents predicted probabilities.

### Activation Function

The sigmoid activation function maps values to the interval between 0 and 1.

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

### Loss Function

The application uses Binary Cross-Entropy (BCE) to measure the difference between actual outputs and predicted probabilities.

```text
Loss = -mean(
    y * log(p) +
    (1 - y) * log(1 - p)
)
```

### Learning Algorithm

The network learns using gradient-descent-based backpropagation.

```text
Initialize weights and biases
          ↓
    Forward propagation
          ↓
       Prediction
          ↓
     Calculate loss
          ↓
     Backpropagation
          ↓
    Calculate gradients
          ↓
      Update weights
          ↓
    Repeat for epochs
```

The weight update follows:

```text
W = W - learning_rate × gradient
```

## 🛠️ Technology Stack

- **Programming Language:** Scilab
- **Development Environment:** Scilab 2026.1.0
- **GUI:** Scilab graphical interface functionality
- **Numerical Computation:** Matrix operations and mathematical functions
- **Visualization:** Scilab plotting and graph visualization

The application uses Scilab's built-in functionality and does not require an external machine learning library or a GPU.

## 📂 Project Structure

```text
NEURO_LAB/
│
├── main.sce
├── gui.sci
├── callbacks.sci
├── neural_network.sci
│
├── Source_Code/
├── Screenshots/
│   ├── main_gui.png
│   ├── trained_network.png
│   └── prediction_results.png
│
├── Documentation/
│   ├── README.md
│   └── Problem_Statement.pdf
│
└── Demo/
    └── NeuroLab_Demo.mp4
```

The root-level source files are the core application files. The additional folders represent the recommended final submission structure; include them if they exist in your repository.

| File | Purpose |
|---|---|
| `main.sce` | Loads project files and launches the application. |
| `gui.sci` | Defines the graphical user interface. |
| `callbacks.sci` | Handles GUI interactions, training, predictions, and visualization. |
| `neural_network.sci` | Contains neural-network-related functionality. |

## ⚙️ Installation and Setup

### Prerequisites

- A computer running Windows, Linux, or macOS.
- Scilab 2026.1.0 or a compatible version.
- All four core project files.

### Steps to Run

**1. Install Scilab**

Download and install Scilab from the official website:

https://www.scilab.org/

**2. Download the project**

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Alternatively, download the repository as a ZIP file and extract it.

**3. Open Scilab**

Launch Scilab and set the current working directory to the folder containing the project files.

**4. Run the application**

Open `main.sce` and execute it.

The NEURO LAB graphical interface should launch.

**5. Train a neural network**

For the example configuration, select:

| Parameter | Example Value |
|---|---|
| Dataset | XOR |
| Learning Rate | 0.5 |
| Epochs | 3000 |
| Hidden Neurons | 4 |

Click **TRAIN NETWORK** to begin training.

**6. Explore the results**

Inspect the loss graph, neural network visualization, prediction table, accuracy, and final loss. Use the network inspection and mathematics features to explore the model further.

## 📊 Sample Output

An illustrative example from the project documentation:

```text
Dataset:        AND
Learning Rate:  0.5
Hidden Neurons: 4
Epochs:         3000

Accuracy:       100%
Final Loss:     0.0021
```

*Note: These are example results documented for the project, not a guarantee of performance on every run. Exact results may vary because of random weight initialization and training settings.*

## 👥 Intended Users

NEURO LAB is designed for:

- Students beginning their AI/ML journey.
- Computer Science and Artificial Intelligence students.
- Beginners learning neural networks.
- Faculty members demonstrating neural network concepts.
- Anyone interested in understanding the fundamentals of model training.

## 🚀 Advantages

- Beginner-friendly and interactive learning environment.
- No external dataset required.
- Demonstrates actual neural network training.
- Allows experimentation with training parameters.
- Provides visual feedback through graphs and network diagrams.
- Connects mathematical operations with model predictions.
- Runs without requiring a GPU.

## ⚠️ Current Limitations

- Supports only the XOR, AND, and OR logical datasets.
- Designed for binary classification.
- Uses sigmoid activation in the current implementation.
- Uses a small neural network and is not intended for large-scale real-world datasets.
- Does not currently provide the capabilities of a full deep-learning framework.

## 🔮 Future Enhancements

Potential improvements include:

- Support for ReLU and Tanh activation functions.
- Custom dataset uploads.
- Multiple hidden layers.
- Decision-boundary visualization.
- Real-time training animation.
- Confusion matrix and additional evaluation metrics.
- Model export and import.
- More interactive weight visualization.
- Additional machine learning demonstrations.


## 🏆 Hackathon Submission

Developed as part of the **Scilab GUIVerse Hackathon 2026**, organized by the **FOSSEE Project, IIT Bombay**.

The project explores how interactive GUI-based tools can make AI and Machine Learning concepts more accessible to learners.

## 📚 References

- [Scilab Official Website](https://www.scilab.org/)
- [Scilab Documentation](https://help.scilab.org/)
- [Scilab GUI Hackathon — Final Submission](https://scilab.in/gui-hackathon/final-submission)

## 👩‍💻 Author

**Megha Rajeev**

AI Engineering Student | Artificial Intelligence & Machine Learning

---

*NEURO LAB — Learn neural networks by experimenting, visualizing, and understanding how they learn.*
