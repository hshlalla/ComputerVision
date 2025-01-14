import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from models import vgg
ngpu = 1                
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
  
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            normalize,
    ])

    if test:
        dataset = datasets.CIFAR100(
          root=data_dir, train=False,
          download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR100( root=data_dir, train=True, download=True, transform=transform)

    valid_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform,)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)

# flower 102 dataset 
train_loader, valid_loader = data_loader(data_dir='./data',
                                         batch_size=64)

test_loader = data_loader(data_dir='./data',
                              batch_size=64,
                              test=True)

num_classes = 100
num_epochs = 20




batch_size = 4096
learning_rate = 0.001

model = vgg.VGG11(ngpu,num_classes).to(device)
# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = 0.005)  


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set model to training mode
    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate training loss and accuracy
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
    train_accuracy = 100 * train_correct / train_total
    train_loss /= len(train_loader)

    print('Epoch [{}/{}], Training Loss: {:.4f}, Training Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, train_loss, train_accuracy))

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            del images, labels, outputs

    val_accuracy = 100 * val_correct / val_total
    val_loss /= len(valid_loader)

    print('Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
          .format(val_loss, val_accuracy))


if __name__ == "__main__":
    pass