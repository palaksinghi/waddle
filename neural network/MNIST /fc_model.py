import torch
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()

        self.hidden_layers = nn.ModuleList()

        # First hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_layers[0]))

        # Remaining hidden layers
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            )

        # Output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))

        x = F.log_softmax(self.output(x), dim=1)

        return x


def train(model, trainloader, testloader, criterion, optimizer, epochs=5):

    train_losses = []
    test_losses = []

    for e in range(epochs):

        running_loss = 0

        model.train()

        for images, labels in trainloader:

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        model.eval()

        test_loss = 0
        accuracy = 0

        with torch.no_grad():

            for images, labels in testloader:

                log_ps = model(images)

                loss = criterion(log_ps, labels)

                test_loss += loss.item()

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += equals.float().mean().item()

        model.train()

        train_loss = running_loss / len(trainloader)
        valid_loss = test_loss / len(testloader)
        valid_accuracy = accuracy / len(testloader)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)

        print(
            f"Epoch {e+1}/{epochs} "
            f"Training Loss: {train_loss:.3f} "
            f"Validation Loss: {valid_loss:.3f} "
            f"Accuracy: {valid_accuracy*100:.2f}%"
        )

    return train_losses, test_losses