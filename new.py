import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])


train_data = datasets.MNIST("", train=True, download=True, transform=transform)
test_data = datasets.MNIST("", train=False, download=True, transform=transform)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=11, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=11, shuffle=False)

for data in train_loader:

    break
plt.imshow(data[0][2].view(28, 28))
print(data[1][2])
