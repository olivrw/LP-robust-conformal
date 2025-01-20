import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        
        # After two 2x2 pools (each halving dimensions), 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)            # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

if __name__ == "__main__":
    # 1) Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Define MNIST transforms
    mean, std = (0.1307,), (0.3081,)
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 3) Load datasets and create loaders
    train_dataset = torchvision.datasets.MNIST(
        root='./datasets',
        train=True,
        download=False,
        transform=train_transforms
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./datasets',
        train=False,
        download=False,
        transform=test_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 4) Define the CNN architecture


    model = SimpleCNN().to(device)

    # 5) Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 6) Training function
    def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 7) Evaluation function
    def evaluate(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    # 8) Execute training
    train_model(model, train_loader, criterion, optimizer, num_epochs=5)

    # 9) Check final accuracy
    test_acc = evaluate(model, test_loader)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # 10) Save model checkpoint
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Trained model saved as 'mnist_cnn.pth'")
