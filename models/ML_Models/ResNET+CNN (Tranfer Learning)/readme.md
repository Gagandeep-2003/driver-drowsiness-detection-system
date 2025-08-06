# Eye Drowsiness Detection using ResNet50

A deep learning project for detecting driver drowsiness by classifying eye states as "awake" or "sleepy" using transfer learning with ResNet50.

## 🎯 Project Overview

This project implements a binary classification model to detect drowsiness in drivers by analyzing eye images. The model achieves **98.5% validation accuracy** using a pre-trained ResNet50 backbone with custom classification layers.

### Key Features
- **Transfer Learning**: Utilizes pre-trained ResNet50 from ImageNet
- **Binary Classification**: Classifies eyes as "awake" or "sleepy"
- **High Accuracy**: Achieves 98.5% validation accuracy
- **GPU Optimized**: Configured for Google Colab with GPU acceleration
- **Production Ready**: Saved model ready for deployment

## 📊 Model Performance

| Metric | Training | Validation |
|--------|----------|------------|
| **Accuracy** | 97.76% | 98.50% |
| **Loss** | 0.0600 | 0.0415 |

## 🗂️ Dataset

The project uses the **MRL Eye Dataset** from Kaggle containing:
- **Training Set**: 50,937 images (80/20 split for train/validation)
- **Validation Set**: 16,980 images  
- **Classes**: 2 (awake, sleepy)
- **Image Size**: 224×224 pixels

**Dataset Source**: [MRL Eye Dataset on Kaggle](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset)

## 🏗️ Model Architecture

```
ResNet50 (Pre-trained, Frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(2, activation='softmax')
```

### Model Specifications
- **Base Model**: ResNet50 (ImageNet pre-trained, frozen)
- **Input Shape**: (224, 224, 3)
- **Output**: 2 classes (awake/sleepy)
- **Total Parameters**: ~25M (only ~260K trainable)
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Google Colab (recommended) or local GPU environment
- Kaggle API credentials

### Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/eye-drowsiness-detection.git
cd eye-drowsiness-detection
```

2. **Install dependencies**:
```bash
pip install tensorflow matplotlib kagglehub
```

3. **Run in Google Colab**:
   - Upload `Drowsiness.ipynb` to Google Colab
   - Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU
   - Run all cells

### Usage

1. **Download Dataset**:
```python
import kagglehub
path = kagglehub.dataset_download("akashshingha850/mrl-eye-dataset")
```

2. **Train the Model**:
```python
# Configure paths
train_folder_path = '/path/to/train'
val_folder_path = '/path/to/val'

# Train model (5 epochs)
history = model.fit(train_ds, validation_data=val_ds, epochs=5)
```

3. **Save the Model**:
```python
model.save('/content/saved_model/resnet.keras')
```

4. **Load and Use for Prediction**:
```python
import tensorflow as tf
loaded_model = tf.keras.models.load_model('/path/to/resnet.keras')
prediction = loaded_model.predict(new_image)
```
## 🔬 Training Details

### Hyperparameters
- **Image Size**: 224×224 pixels
- **Batch Size**: 16
- **Epochs**: 5
- **Learning Rate**: Adam default (0.001)
- **Validation Split**: 20%
- **Data Augmentation**: None (can be added for improvement)

### Training Results
```
Epoch 1/5: val_accuracy: 0.9753
Epoch 2/5: val_accuracy: 0.9756  
Epoch 3/5: val_accuracy: 0.9761
Epoch 4/5: val_accuracy: 0.9806
Epoch 5/5: val_accuracy: 0.9850
```
## 🔧 Customization & Improvements

### Potential Enhancements
- **Data Augmentation**: Add rotation, brightness, contrast adjustments
- **Fine-tuning**: Unfreeze top layers of ResNet50 for better accuracy
- **Real-time Processing**: Optimize for webcam/camera input
- **Multi-class**: Extend to detect different levels of drowsiness
- **Ensemble Methods**: Combine multiple models for better performance

### Fine-tuning Example
```python
# Unfreeze top layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Use lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## 📈 Performance Analysis

The model shows excellent performance with:
- **No Overfitting**: Validation accuracy consistently higher than training
- **Quick Convergence**: Reaches high accuracy within 5 epochs
- **Stable Training**: Consistent improvement across epochs

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features or improvements
- Submit pull requests
- Share your results and modifications

## 🙏 Acknowledgments

- **Dataset**: MRL Eye Dataset by akashshingha850 on Kaggle
- **Base Model**: ResNet50 from TensorFlow/Keras
- **Platform**: Google Colab for providing free GPU resources
