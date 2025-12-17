# Text-to-Image Generation with Stable Diffusion & CLIP

A comprehensive implementation of fine-tuned Stable Diffusion v1.4 on the COCO dataset, leveraging CLIP for semantic text encoding and advanced evaluation metrics.

## 🎯 Project Overview

This project implements a complete text-to-image generation pipeline with:
- **Fine-tuning** Stable Diffusion on domain-specific COCO dataset
- **CLIP integration** for advanced text encoding and semantic understanding
- **Multi-scheduler evaluation** (DPM++, PNDM, Euler, DDIM)
- **Comprehensive metrics** (FID, Inception Score)
- **Production-ready deployment** with custom generation capabilities

## ✨ Key Features

- 🎨 **Advanced Image Generation**: State-of-the-art text-to-image synthesis
- 🔧 **Fine-tuned Models**: Custom training on COCO dataset with 1000+ image-caption pairs
- 📊 **Evaluation Suite**: FID, Inception Score, and semantic similarity metrics
- ⚡ **Multiple Schedulers**: Compare DPM++, PNDM, Euler, and DDIM samplers
- 🎛️ **Guidance Scale Tuning**: Optimize for quality vs. prompt adherence
- 📈 **Comprehensive Logging**: Track training progress and model performance

## 🏗️ Architecture

```
Text Input → CLIP Text Encoder → Stable Diffusion U-Net → VAE Decoder → Generated Image
                                         ↓
                                  Multiple Schedulers
                                  (DPM++/PNDM/Euler/DDIM)
```

## 📋 Requirements

### Core Dependencies
```
torch>=2.5.1
diffusers>=0.32.2
transformers>=4.49.0
accelerate>=0.20.0
jupyter>=1.0.0
```

### Additional Libraries
```
numpy
pandas
pillow
matplotlib
scipy
tqdm
pycocotools
```

### Evaluation Metrics
```
pytorch-fid
torch-fidelity
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shruthik99/Text-to-Image-generation.git
cd Text-to-Image-generation
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

## 💻 Usage

### Running the Notebook

1. Open `IE_7615_Generative_Project_Final.ipynb` in Jupyter
2. Run cells sequentially from top to bottom
3. The notebook is organized into sections:
   - Environment Setup
   - Data Preprocessing
   - CLIP Embedding Generation
   - Model Fine-tuning
   - Scheduler Comparison
   - Evaluation Metrics
   - Custom Image Generation

### Quick Generation (After Training)

```python
from diffusers import StableDiffusionPipeline
import torch

# Load the fine-tuned model
pipe = StableDiffusionPipeline.from_pretrained(
    "./fine_tuned_model",
    torch_dtype=torch.float16
).to("cuda")

# Generate image
prompt = "A serene sunset over mountain peaks with vibrant colors"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("generated_image.png")
```

## 📊 Model Performance

| Scheduler | FID Score ↓ | Inception Score ↑ | Generation Time |
|-----------|-------------|-------------------|-----------------|
| DPM++ 2M  | 28.4        | 3.82              | 4.2s            |
| PNDM      | 31.7        | 3.64              | 5.8s            |
| Euler     | 29.8        | 3.71              | 3.9s            |
| DDIM      | 30.2        | 3.68              | 4.5s            |

*Tested on NVIDIA A100 GPU with 50 inference steps*

## 📁 Project Structure

```
Text-to-Image-generation/
├── IE_7615_Generative_Project_Final.ipynb  # Main notebook
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies
├── .gitignore                              # Git ignore rules
├── data/
│   ├── coco_captions/                      # COCO dataset captions
│   └── processed/                          # Preprocessed data
├── models/
│   ├── fine_tuned_model/                   # Fine-tuned SD model
│   └── checkpoints/                        # Training checkpoints
└── outputs/
    ├── generated_images/                   # Generated samples
    └── evaluation_results/                 # Metrics and logs
```

## 🔬 Evaluation Metrics

### Fréchet Inception Distance (FID)
- Measures distribution similarity between generated and real images
- Lower is better (typical range: 20-50)

### Inception Score (IS)
- Evaluates quality and diversity of generated images
- Higher is better (typical range: 2-5)

### CLIP Similarity
- Semantic alignment between text prompts and generated images
- Range: 0-1 (higher indicates better text-image correspondence)

## 🎓 Technical Details

### Fine-tuning Strategy
- **Base Model**: Stable Diffusion v1.4
- **Dataset**: COCO 2017 (filtered subset)
- **Training Steps**: 1000+ iterations
- **Learning Rate**: 1e-5 with cosine scheduling
- **Batch Size**: 4 (with gradient accumulation)

### Text Encoding
- **Model**: CLIP ViT-L/14
- **Embeddings**: 768-dimensional vectors
- **Max Tokens**: 77 tokens per prompt

### Image Generation
- **Resolution**: 512x512 pixels
- **Inference Steps**: 20-50 (configurable)
- **Guidance Scale**: 7.5 (optimal for balance)

## 📈 Notebook Sections

1. **Environment Setup** - Package installation and imports
2. **Data Preprocessing** - COCO dataset filtering and preparation
3. **CLIP Embeddings** - Text encoding generation
4. **Model Fine-tuning** - Stable Diffusion training loop
5. **Scheduler Comparison** - Evaluate different sampling methods
6. **Metrics Calculation** - FID and Inception Score computation
7. **Custom Generation** - Interactive image generation interface

## 🛠️ Advanced Configuration

### Scheduler Customization
```python
from diffusers import DPMSolverMultistepScheduler

# Use DPM++ for faster generation
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)
```

### Memory Optimization
```python
# Enable attention slicing for lower VRAM usage
pipe.enable_attention_slicing()

# Enable VAE slicing
pipe.enable_vae_slicing()
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stable Diffusion** by Stability AI
- **CLIP** by OpenAI
- **Diffusers Library** by Hugging Face
- **COCO Dataset** by Microsoft

## 📧 Contact

**Shruthi Kashetty**
- GitHub: [@Shruthik99](https://github.com/Shruthik99)
- Project: [Text-to-Image-generation](https://github.com/Shruthik99/Text-to-Image-generation)

## 🔗 References

1. [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
2. [CLIP Paper](https://arxiv.org/abs/2103.00020)
3. [COCO Dataset](https://cocodataset.org/)
4. [Diffusers Documentation](https://huggingface.co/docs/diffusers/)

---

⭐ **Star this repository** if you find it helpful!
