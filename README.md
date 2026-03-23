#  DementAI – Alzheimer's Disease Detection from Brain MRI using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project aims to classify different stages of **Alzheimer's disease** (and non-demented cases) from **brain MRI images** using modern convolutional neural networks and explainability techniques.

**4-class classification:**
- Non-Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

##  Key Features
- State-of-the-art model: **EfficientNet-B3** (pre-trained)
- Strong data augmentation & training pipeline
- **Grad-CAM** visualization for model interpretability (shows which brain regions the model focuses on)
- Simple **Streamlit** web interface for uploading and testing images
- Clean training script with validation & model checkpointing

## Sample Visual Results

**Brain MRI comparison – Healthy vs. Alzheimer's (noticeable atrophy in advanced cases):**

<grok-card data-id="b7813f" data-type="image_card" data-plain-type="render_searched_image"  data-arg-size="LARGE" ></grok-card>



<grok-card data-id="dd1501" data-type="image_card" data-plain-type="render_searched_image"  data-arg-size="LARGE" ></grok-card>


**Grad-CAM examples – Highlighting regions the model pays attention to:**

<grok-card data-id="e637ce" data-type="image_card" data-plain-type="render_searched_image"  data-arg-size="LARGE" ></grok-card>


## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/DementAI.git
cd DementAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Train the model yourself
python train.py

# 4. Run the demo app
streamlit run app.py
