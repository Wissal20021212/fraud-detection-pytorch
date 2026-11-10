import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset , DataLoader

from model import FraudDetectionModel
from prepare_data import load_data

# 1. Charger les données
X_train, X_val, X_test, y_train, y_val, y_test = load_data()

# 2. Créer les DataLoaders
batch_size = 2048

train_dataset = TensorDataset(X_train, y_train.unsqueeze(1))
val_dataset = TensorDataset(X_val, y_val.unsqueeze(1))

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 3. Initialiser le modèle
model = FraudDetectionModel()

# 4. Définir la fonction de perte
num_fraud = (y_train == 1).sum()
num_normal = (y_train == 0).sum()
pos_weight = num_normal / num_fraud

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

# 5. Définir l'optimiseur
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

best_val_loss = float("inf")
patience = 3
counter = 0


# 6. Entraînement
num_epochs = 20

for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)
    # 7. Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(val_loader.dataset)

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0

        torch.save(model.state_dict(), "fraud_model.pth")

    else:
        counter += 1

        if counter >= patience:
            print("Early stopping!")
            break

print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )


# 8. Sauvegarder le modèle
torch.save(model.state_dict(), "fraud_model.pth")

print("Training completed!")


