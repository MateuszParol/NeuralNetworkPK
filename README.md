# Neural Network - Character Recognition

A clean, modern neural network application for recognizing handwritten characters (0-9, A-Z).

## Features

- **Interactive Drawing**: Draw characters with your mouse
- **Real-time Recognition**: Instant character prediction with confidence scores
- **Network Visualization**: Live visualization of neural network activity
- **Training on EMNIST**: Load and train on the EMNIST dataset
- **Modern UI**: Clean, minimalist interface with Material Design

## Requirements

```bash
pip install numpy pandas
```

- Python 3.7+
- numpy (required)
- pandas (optional, for training)
- tkinter (built-in)

## Quick Start

```bash
python neuralnetwork.py
```

## Usage

### Drawing & Recognition

1. **Draw** a character on the white canvas (left panel)
2. **Release** the mouse button to see the prediction
3. **Clear** to reset and draw again

### Training

1. Click **"Load & Train"**
2. Select your EMNIST CSV file (`emnist-balanced-train.csv`)
3. Wait for training to complete (~15,000 samples)
4. Test the trained network by drawing characters

### Visualization

The center panel shows the neural network in action:
- **Left**: Input layer (20 representative nodes of 784 total)
- **Center**: Hidden layer (32 visible nodes of 128 total)
- **Right**: Output layer (36 nodes for 0-9, A-Z)
- **Lines**: Active connections (green = positive weights, red = negative)

## Architecture

```
Input Layer:  784 neurons (28x28 pixels)
      ↓
Hidden Layer: 128 neurons (sigmoid activation)
      ↓
Output Layer: 36 neurons (softmax activation)
```

### Key Features

- **Xavier Initialization**: For better convergence with sigmoid
- **Softmax Output**: Proper probability distribution for multi-class classification
- **Learning Rate**: 0.05 (balanced for stability and speed)
- **Weight Clipping**: Prevents exploding gradients

## Code Structure (318 lines)

- **Lines 1-16**: Imports and configuration
- **Lines 18-64**: `NeuralNetwork` class (forward pass, backpropagation)
- **Lines 66-317**: `App` class (UI, visualization, training)

## EMNIST Dataset

The application expects CSV format:
```
label, pixel_0, pixel_1, ..., pixel_783
5, 0, 0, 15, ..., 0
10, 0, 23, 45, ..., 12
```

- Column 0: Label (0-35 used)
- Columns 1-784: Pixel values (0-255)
- No header

**Label mapping**:
- 0-9: Digits
- 10-35: Letters A-Z

Download EMNIST: [NIST Website](https://www.nist.gov/itl/products-and-services/emnist-dataset)

## Troubleshooting

### Import Error: No module named 'numpy'
```bash
pip install numpy
```

### Import Error: No module named 'pandas'
```bash
pip install pandas
```
(Only needed for training)

### tkinter not found (Linux)
```bash
sudo apt install python3-tk
```

### Network always predicts same character

This happens with an untrained network. Solution:
1. Load EMNIST dataset
2. Click "Load & Train"
3. Wait for training to complete
4. Try drawing again

## Design Philosophy

- **Clean & Modern**: Material Design inspired
- **Minimalist**: No unnecessary elements
- **Functional**: Every element serves a purpose
- **Readable**: Clear typography and spacing

## Color Palette

- Background: `#F5F5F5` (Light gray)
- Panels: `#FFFFFF` (White)
- Primary: `#2196F3` (Blue)
- Success: `#4CAF50` (Green)
- Error: `#F44336` (Red)
- Text: `#333333` (Dark gray)

## License

MIT License - Free to use and modify.
