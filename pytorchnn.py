#Import dependencies
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#get  data our
train = datasets.MNIST(root='data', download=True, train=True, transform=ToTensor())
datasets = DataLoader(train, 32)
#1,28,28 - classes 0-9

#Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __int__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )
    def forward(self, x):
        return self.model(x)

#Instance of th neural network, loss, optimizer
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

#Training flow
if __name__ == "__name__":
    for epoch in range(10): #train for 10 epochs
        for batch in datasets:
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            #Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f'Epoch:{epoch} loss is {loss.item()}')

    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)



