import torch
from model import create_model
from dataset import load_data
from metrics import evaluate

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = load_data()
    model = create_model(num_classes=len(classes)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}")
        evaluate(model, test_loader, device)

    torch.save(model.state_dict(), "outputs/model.pth")

if __name__ == "__main__":
    train()
