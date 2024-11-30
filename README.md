# F1 Car Classification Using PyTorch

This project demonstrates building, training, and evaluating a deep learning model to classify images of Formula 1 (F1) cars using PyTorch. It covers essential steps like data loading, preprocessing, model creation, training, and validation. The code is modular, enabling easy customization and extension.

---

## 📚 Table of Contents

1. [About the Project](#about-the-project)
2. [Technologies Used](#technologies-used)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Data Preparation](#data-preparation)
6. [Model Architecture](#model-architecture)
7. [Training the Model](#training-the-model)
8. [Evaluation and Testing](#evaluation-and-testing)
9. [Customization](#customization)


---

## 🔍 About the Project

This project uses PyTorch to classify images of Formula 1 (F1) cars based on their visual features. The dataset consists of images representing different F1 teams or models. The focus is on building a `Sequential` model and handling image data efficiently using a custom dataset class. The code is designed to be flexible, allowing for modifications based on different datasets or model architectures.

---

## 🛠️ Technologies Used

- **PyTorch**
- **Torchvision**
- **Pandas**
- **Scikit-image**
- **Python 3.x**

---

## 📂 Project Structure

- **Custom Dataset Class:** Defines how to load and preprocess image data.
- **Model Training:** Implements the training loop, loss calculation, and optimization steps.
- **Data Loading:** Uses PyTorch's `DataLoader` for efficient batching and shuffling.
- **Model Evaluation:** Provides methods to validate model performance on test data.

---

## ⚙️ Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies:**
   ```bash
   pip install torch torchvision pandas scikit-image
   ```

3. **Prepare Your Dataset:**
   - Create a CSV file containing the paths to images and their corresponding labels (team names or car models).
   - Organize your image data into structured folders.

---

## 📊 Data Preparation

The project includes a custom dataset class to handle image data:

```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        label = torch.tensor(int(self.img_labels.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        return image, label
```

---

## 🏎️ Model Architecture

A PyTorch `Sequential` model is customized to classify F1 car images:

```python
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(32 * 16 * 16, num_classes)  # 'num_classes' for F1 teams/models
)
```

---

## 🚀 Training the Model

Define the training loop, optimizer, and loss function:

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 📈 Evaluation and Testing

Evaluate the model's performance using a validation set:

```python
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        # Calculate accuracy or other metrics
```

---

## 🔧 Customization

- **Model:** Modify the `Sequential` architecture to suit different tasks.
- **Data:** Adjust the dataset class for new formats or preprocessing steps.
- **Hyperparameters:** Tune learning rates, batch sizes, and epochs for better performance.
