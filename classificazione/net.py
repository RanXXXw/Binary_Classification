import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from pytorch_model_summary import summary

class Net(nn.Module):
    """
    Semplice rete di classificazione immagini.
    Deriva da nn.Module di pytorch.
    """    
    def __init__(self, classes : list[str]) -> None:
        super(Net, self).__init__()

      # Primo strato convoluzionale
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # input, output, filtro
        self.relu1 = nn.ReLU()  # non linearità
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # evita overfitting

        # Secondo strato convoluzionale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Strato completamente connesso
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # L'output dei conv è 64x64 dopo 2 pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 3) 

    def forward(self, x):
         # Passaggio in avanti attraverso i livelli convoluzionali
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Ridimensionamento per il passaggio attraverso i livelli completamente connessi
        x = x.view(-1, 32 * 64 * 64)


        # Passaggio attraverso i livelli completamente connessi
        x = self.relu3(self.fc1(x))  
        x = self.fc2(x)
          
        return x
    
if __name__ == '__main__':

    # Crea l'oggetto che rappresenta la rete.
    # Fornisce le classi.
    n = Net(['a', 'b', 'c'])

    # Crea cartella se non esiste
    file_path = './out/model_state_dict.pth'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Salva i parametri addestrati della rete.
    torch.save(n.state_dict(), file_path)
    
    # Salva l'intero modello.
    torch.save(n, file_path)
    
    # Stampa informazioni generali sul modello.
    print(n)

    # Stampa i parametri addestrabili.
    # for name, param in n.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    # Stampa un recap del modello.
    print(summary(n, torch.ones(size=(1, 3, 256, 256))))