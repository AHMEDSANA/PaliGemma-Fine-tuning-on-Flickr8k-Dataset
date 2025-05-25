# 🎨 PaliGemma Fine-Tuning for Image Captioning on Flickr8k Dataset

> Fine-tune Google's PaliGemma vision-language model on the Flickr8k dataset for image captioning tasks.

## 🌟 Overview

PaliGemma is a powerful vision-language model that combines image understanding with text generation capabilities. This project demonstrates how to fine-tune the model on a custom dataset (Flickr8k) to improve its image captioning performance.

## ✨ Features

- Fine-tuning PaliGemma on Flickr8k dataset
- Custom training loop with progress tracking
- Model saving and loading utilities
- Batch inference for testing
- Visual result rendering
- JAX/Flax implementation with efficient sharding

## 📋 Requirements

```bash
pip install ml_collections
pip install tensorflow
pip install sentencepiece
pip install jax
pip install sklearn
pip install pandas
pip install numpy
pip install pillow
pip install kagglehub  # For downloading model weights
```

**Additional Requirements:**
- Flickr8k dataset from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k
- Google's big_vision repository (automatically cloned)
- PaliGemma model weights (automatically downloaded via kagglehub)
- PaliGemma tokenizer (automatically downloaded from Google Cloud Storage)

## 📁 Project Structure

```
├── finetune-paligemma/
│   ├── finetune-paligemma.ipynb              # Training and testing script 
├── PaliGemma_test_script.py                  # Standalone test script
└── README.md
```

## 📊 Dataset

The project uses the Flickr8k dataset from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k

**Dataset Details:**
- 8,000 images with human-annotated captions
- Multiple captions per image (5 captions per image)
- Diverse scene descriptions covering people, animals, activities, and outdoor scenes
- High-quality annotations suitable for vision-language tasks

### Dataset Structure
```
flickr8k/
├── captions.txt       # CSV file with image-caption pairs
└── Images/            # Directory containing 8,000 JPEG images
    ├── 1000268201_693b08cb0e.jpg
    ├── 1001773457_577c3a7d70.jpg
    └── ...
```

### Dataset Setup
1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k
2. Extract to your desired location
3. Update the `DATA_DIR` path in the scripts to point to your dataset location

## 🏋️ Training

The training script (`finetune-paligemma.ipynb`) includes:

### Key Features:
- **Model Configuration**: PaliGemma 3B parameter model with specific image and language settings
- **Fine-tuning Strategy**: Only attention layers are trainable, keeping other parameters frozen
- **Data Processing**: Robust image preprocessing and tokenization
- **Training Loop**: 10 epochs with batch size of 10
- **Progress Tracking**: Real-time loss monitoring and time estimation
- **Model Saving**: Custom parameter saving with fallback mechanisms

### Training Parameters:
| Parameter | Value |
|-----------|-------|
| **Learning Rate** | 5e-5 |
| **Batch Size** | 10 |
| **Epochs** | 10 |
| **Sequence Length** | 128 tokens |
| **Image Size** | 224x224 pixels |
| **Dataset Size** | 1,500 samples (1,200 train / 300 validation) |

### Usage:
```python
# Download Flickr8k dataset from Kaggle
# https://www.kaggle.com/datasets/adityajn105/flickr8k

# Update DATA_DIR path in the script to point to your dataset location
DATA_DIR = "/path/to/flickr8k"

# The training script automatically:
# 1. Downloads PaliGemma model weights via kagglehub
# 2. Downloads tokenizer from Google Cloud Storage
# 3. Loads and preprocesses Flickr8k dataset (1,500 samples)
# 4. Fine-tunes only attention layers
# 5. Saves the trained model to /kaggle/working/PaliGemma_Fine_Tune_Flickr8k/ 
#    (update location if running locally)
```

## 🔮 Inference

The inference scripts (`finetune-paligemma.ipynb` and `PaliGemma_test_script.py`) provide:

### Features:
- Load fine-tuned model weights
- Batch processing of test images
- Caption generation with customizable sampling
- Visual result rendering
- Robust error handling

### Usage:
```python
# Ensure the fine-tuned model exists at the specified path
MODEL_PATH = "/kaggle/working/PaliGemma_Fine_Tune_Flickr8k/paligemma_flickr8k.params.f16.npz"
# Update the location to the actual path if running locally

# Update paths in the script if needed
DATA_DIR = "/path/to/flickr8k"
IMAGES_DIR = "/path/to/flickr8k/Images"

# Run inference
python PaliGemma_test_script.py
```

## 🏗️ Model Architecture

| Component | Specification |
|-----------|---------------|
| **Base Model** | PaliGemma 3B |
| **Vision Encoder** | SigLIP-So400m/14 |
| **Language Model** | Custom LLM with 257,152 vocabulary |
| **Fine-tuning** | Attention layers only |
| **Input Resolution** | 224x224 |
| **Max Sequence Length** | 128 tokens |

## 🔄 Training Process

1. **Data Loading**: Flickr8k images and captions are loaded and preprocessed (1000 images used due to system constraints - you can use full 8000 with increased epochs for better results)
2. **Model Setup**: PaliGemma model is initialized with pretrained weights
3. **Parameter Masking**: Only attention layers are marked as trainable
4. **Training Loop**: Model is trained for 10 epochs with progress monitoring
5. **Validation**: Model performance is evaluated on held-out validation set
6. **Model Saving**: Trained parameters are saved for later inference

## 📈 Results

### Training Metrics
- Training completed over 10 epochs
- Loss reduction observed during training
- Validation performed on 20% held-out data

### Sample Outputs

| Image | Generated Caption |
|-------|-------------------|
| ![image](https://github.com/user-attachments/assets/78a9af42-f308-46ee-a7c3-40b6ee5d79f5) | a biker is doing a wheelie on a yellow motorcycle |
| ![image](https://github.com/user-attachments/assets/c4c3d931-6ea2-4ff9-b998-0b5f6f2b37b6) | three women dressed in green for the parade |
| ![image](https://github.com/user-attachments/assets/819682d1-1c2e-4ffd-9a3c-f54f9aab7208) | a brown and white dog standing in the water |

## 🔧 Implementation Details

### Key Components:

**1. Image Preprocessing:**
- Resize to 224x224
- Normalize to [-1, 1] range
- Handle grayscale to RGB conversion

**2. Text Processing:**
- SentencePiece tokenization
- Proper masking for training
- EOS token handling

**3. Training Strategy:**
- Selective parameter freezing
- Gradient accumulation
- Memory-efficient sharding

**4. Inference Pipeline:**
- Batch processing
- Greedy decoding
- Post-processing cleanup

## ⚡ Performance Considerations

- **Memory Usage**: Efficient parameter sharding for large model
- **Training Speed**: ~120 steps per epoch with batch size 10
- **Inference Speed**: Batch processing for efficient caption generation
- **Model Size**: Fine-tuned model maintains original parameter count

## 🛠️ Troubleshooting

### Common Issues:

| Issue | Solution |
|-------|----------|
| **Memory Errors** | Reduce batch size or use gradient checkpointing |
| **Loading Errors** | Ensure proper model path and file permissions |
| **Dataset Issues** | Verify Flickr8k dataset structure and file paths |
| **JAX Issues** | Ensure proper JAX installation and device configuration |

**Additional Solutions:**
- Custom model saving/loading functions included for robustness
- Fallback mechanisms for parameter serialization
- Error handling for corrupted images
- Progress tracking for long training runs

## 🚀 Future Work

- [ ] Experiment with different learning rates
- [ ] Add BLEU/ROUGE evaluation metrics
- [ ] Implement beam search decoding
- [ ] Fine-tune on additional datasets
- [ ] Add multi-GPU training support
- [ ] Implement gradual unfreezing strategy

## 🙏 Acknowledgments

- Google Research for PaliGemma model
- Big Vision team for the implementation framework
- Flickr8k dataset creators
- JAX/Flax development team
