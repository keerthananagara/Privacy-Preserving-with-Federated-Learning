
# 🛡️Privacy-Preserving Neural Networks in Federated Learning

This project presents a Privacy-Preserving Federated Learning (FL) Framework using Scatter-CNN and Rényi Differential Privacy (RDP). It enables decentralized model training with a focus on protecting user data privacy while maintaining high performance in image classification tasks.

## 🚀 Project Overview

- Developed a federated learning system that trains models locally on clients without sharing raw data.
- Integrated differential privacy mechanisms to mitigate information leakage.
- Utilized Scatter CNN for efficient and accurate image classification.
- Evaluated on benchmark datasets: MNIST and FMNIST.

## 🧠 Key Features

- 📊 Accuracy:
  - MNIST: Improved from 94% to 96% with privacy
  - FMNIST: Achieved 82% with privacy
- 🔒 Privacy Techniques:
  - Differential Privacy with calibrated noise (ε = 4.0, δ = 1e-5)
  - Gradient clipping for noise control
- 🔄 Federated Architecture:
  - Trained with 1, 5, and 10 clients
  - Uses FedAvg aggregation for model updates

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**: PyTorch, TorchVision, Kymatio
- **Tools**: Google Colab, Jupyter Notebook, VS Code
- **Frameworks**: Manual Differential Privacy Accounting (custom implementation)

## 🖼️ Datasets Used

- **MNIST**: Handwritten digit images
- **FMNIST**: Fashion item images
- Each dataset split across clients to simulate non-IID settings.

## 📂 Project Structure

├── client/                 # Local model training code
├── server/                 # Global aggregation logic
├── data/                   # MNIST and FMNIST datasets
├── models/                 # Scatter CNN model
├── utils/                  # Privacy functions, preprocessing
└── README.md               # Project documentation

## ✅ Results

- Demonstrated a robust trade-off between privacy and accuracy.
- Personalized training enabled better adaptation to client-specific data.
- Reduced communication overhead by optimizing training rounds.

## 📌 Future Enhancements

- Integration with **Secure Multiparty Computation (SMC)**
- Real-time federated deployment on **IoT/Edge devices**
- Adaptive federated optimization for heterogeneous clients

## 📃 Authors

N.Keerthana  
