# Texture Classification Model

A Python-based deep learning project for classifying texture images using the **ResNeXt-101** convolutional neural network architecture. The model is trained to classify texture patterns efficiently and is paired with a **Gradio** web interface for interactive testing.

## 🧠 Model

This project uses **ResNeXt-101**, a powerful deep convolutional neural network that combines the strengths of ResNet and grouped convolutions to provide high accuracy with efficient computation. It’s pre-trained on ImageNet and fine-tuned for texture classification using a custom dataset.

## 📁 Project Structure
- `main.py`: Loads the dataset, builds, trains, and evaluates the ResNeXt101 model.
- `gradio/app.py`: Provides a Gradio UI for testing the trained model.
- `LICENSE`: Apache 2.0 License.
- `README.md`: Project overview and instructions.

## 🚀 Features
- Texture classification using transfer learning with ResNeXt101.
- Real-time image prediction through an intuitive Gradio interface.
- Modular, easy-to-understand codebase.

## 🛠️ Installation

1. **Clone the repository**
   
git clone https://github.com/noelkenny15/Texture-Classification-Model.git
cd Texture-Classification-Model

2. **Install Required Packages**

pip install -r requirements.txt

3. **Train the Model**

python main.py

4. **Launch Gradio Interface**

python gradio/app.py

