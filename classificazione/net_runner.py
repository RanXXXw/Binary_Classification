import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
from net import Net


class NetRunner():
    """
    Gestisce addestramento e test della rete di classificazione.
    """ 
    def __init__(self, classes : list[str], batch_size : int) -> None:
        """
        Inizializza il gestore e gli attributi necessari all'addestramento.

        Args:
            classes (list[str]): Lista classe del dataset.
            batch_size (int): Campioni per step.
        """        
        self.net = Net(classes)
        self.outpath_sd = './out/trained_model_sd.pth'
        self.outpath = './out/trained_model.pth'
        self.classes = classes
        self.batch_size = batch_size
        self.lr = 0.005
        self.momentum = 0.9 # coefficiente di momentum utilizzato nell'ottimizzatore SGD ,regola l'aggiornamento dei pesi
        self.epochs = 2
 
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), 
                                   lr = self.lr, 
                                   momentum = self.momentum)
        self.writer = SummaryWriter()
      
    def train(self, trainloader : torch.utils.data.DataLoader, validationloader : torch.utils.data.DataLoader, preview : bool = False) -> None:
        """
        Esegue l'addestramento della rete.

        Args:
            trainloader (torch.utils.data.DataLoader): DataLoader per accesso a dati di training.
            preview (bool, optional): Indica se mostrare una anteprima dei dati. Defaults to False.
        """

        # Calcola la dimensione del set di validazione (20% del training set)
        validation_size = int(0.2 * len(trainloader.dataset))
        
        # Ottieni gli indici casuali per il set di validazione
        validation_indices = random.sample(range(len(trainloader.dataset)), validation_size)
        
        # Ottieni gli indici per il set di addestramento (complementare al set di validazione)
        train_indices = [i for i in range(len(trainloader.dataset)) if i not in validation_indices]
        
        # Crea DataLoader per il set di addestramento e il set di validazione
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
        
        trainloader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=self.batch_size, sampler=train_sampler)
        validationloader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=self.batch_size, sampler=validation_sampler)

        if preview:
            self.show_preview(trainloader)

        # Ogni quanto monitorare la funzione di costo.
        step_monitor = 5

        self.losses_x, self.losses_y = [], []
        self.run_losses_x, self.run_losses_y = [], []

        ctr = 0

        best_validation_loss = float('inf')
        
        early_stopping_counter = 0
        patience = 5    

        # Inizializza il writer di TensorBoard
        tb_step = 0
        tb_writer = SummaryWriter()

        # Loop di addestramento per n epoche.
        for epoch in range(self.epochs):

            running_loss = 0.0
            correct_train = 0
            total_train = 0

            total_val = 0  
            correct_val = 0  

            # Stop di addestramento. Dimensione batch_size.
            for i, data in enumerate(trainloader, 0):

                # Le rete entra in modalita' addestramento.
                self.net.train()

                # Per ogni input tiene conto della sua etichetta.
                inputs, labels = data

                # L'input attraversa al rete. Errori vengono commessi.
                # L'input diventa l'output.
                outputs = self.net(inputs)

                # Calcolo della funzione di costo sulla base di predizioni e previsioni.
                loss = self.criterion(outputs, labels)
                
                # I gradienti vengono azzerati.
                self.optimizer.zero_grad()

                # Avviene il passaggio inverso.
                loss.backward()
                
                # Passo di ottimizzazione
                self.optimizer.step()

                # Monitoraggio statistiche.
                running_loss += loss.item()

                if i % step_monitor == 0:
                    self.run_losses_y.append(running_loss / step_monitor)
                    self.run_losses_x.append(ctr)

                    # Aggiorna i valori su TensorBoard
                    tb_writer.add_scalar('Training Loss', running_loss / step_monitor, tb_step)
                    tb_writer.add_images('Training Images', inputs, global_step=tb_step)
                    tb_step += 1

                    print(f'Epoca: {epoch + 1:3d}, Step: {i + 1:5d}] loss: {loss.item():.6f} - running_loss: {(running_loss / step_monitor):.6f}')
                    running_loss = 0.0
                
                self.losses_y.append(loss.item())
                self.losses_x.append(ctr)

                ctr += 1

            # Fase di validazione ad ogni epoca
            self.net.eval()
            with torch.no_grad():
                validation_loss = 0.0

                for i, data in enumerate(validationloader, 0):
                    inputs, labels = data
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    validation_loss += loss.item()
                
                    # Aggiorna la miglior performance e salva il modello se necessario
                    if epoch == 0 or validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        early_stopping_counter = 0
                        torch.save(self.net.state_dict(), self.outpath_sd)
                        torch.save(self.net, self.outpath) 
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter >= patience:
                            print(f'Early stopping at epoch {epoch + 1}, no improvement in validation loss for {patience} epochs.')
                            break

                validation_loss /= len(validationloader)
                
                
            # Calcola accuracy di training per ogni epoca
            self.net.eval()
            with torch.no_grad():
                for data in trainloader:
                    images, labels = data
                    outputs = self.net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

            # Calcola accuracy di validation per ogni epoca
            self.net.eval()
            with torch.no_grad():
                for data in validationloader:
                    images, labels = data
                    outputs = self.net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            train_accuracy = 100 * correct_train // total_train
            val_accuracy = 100 * correct_val // total_val

            print(f'Epoca: {epoch + 1}, Accuracy di Training: {train_accuracy}%, Accuracy di Validation: {val_accuracy}%')
            print('\n')

            # Chiudi il writer di TensorBoard
            tb_writer.close()

        print('Finished Training')
        print('Best Model saved')

        plt.plot(self.losses_x, self.losses_y, label='Loss')
        plt.plot(self.run_losses_x, self.run_losses_y, label='Running Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def test(self, testloader : torch.utils.data.DataLoader, full_test : bool = True, preview : bool = False):
        """
        Esegue un test della rete.

        Args:
            testloader (torch.utils.data.DataLoader): DataLoader per accesso a dati di test.
            preview (bool, optional): Indica se mostrare una anteprima dei dati. Defaults to False.
        """

        if preview:
            self.show_preview(testloader)

        if not full_test:

            dataiter = iter(testloader)
            images, labels = next(dataiter)

            net = Net(self.classes)
            net.load_state_dict(torch.load(self.outpath_sd))
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            gts = [f'{self.classes[labels[j]]:5s}' for j in range(self.batch_size)]
            prs = [f'{self.classes[predicted[j]]:5s}' for j in range(self.batch_size)]
            for i, d in enumerate(zip(gts, prs)):
                gt, pr = d
                print(f'{i+1:2d} - GT vs. Pred.: {"OK" if (gt == pr) else ""} {gt} - {pr}')
        else:

            total, correct = 0, 0
            correct_pred = {classname: 0 for classname in self.classes}
            total_pred = {classname: 0 for classname in self.classes}

            all_labels = []
            all_predicted = []

            net = Net(self.classes)
            net.load_state_dict(torch.load(self.outpath_sd))

            # La rete entra in modalità inferenza.
            net.eval()

            # Non è necessario calcolare i gradienti al passaggio dei dati in rete.
            with torch.no_grad():

                # Cicla i campioni di test, batch per volta.
                for i, data in enumerate(testloader, 0):

                    # Dal batch si estraggono dati ed etichette.
                    images, labels = data

                    # I dati passano nella rete e generano gli output.
                    outputs = net(images)

                    # Dagli output si evince la predizione finale ottenuta.
                    _, predicted = torch.max(outputs.data, 1)

                    # Totali e corretti vengono aggiornati.
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    for label, prediction in zip(labels, predicted):
                        if label == prediction:
                            correct_pred[self.classes[label]] += 1
                        total_pred[self.classes[label]] += 1

                    # Accumula etichette reali e predizioni
                    all_labels.extend(labels.numpy())
                    all_predicted.extend(predicted.numpy())

            print(f'Total network accuracy: {100 * correct // total} %')

            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'Class accuracy: {classname:5s} is {accuracy:.1f} %')

            # Calcola la matrice di confusione e visualizza
            conf_matrix = confusion_matrix(all_labels, all_predicted)
            print("Matrice di confusione:")
            print(conf_matrix)

            # Visualizza un report di classificazione
            class_report = classification_report(all_labels, all_predicted, target_names=self.classes)
            print("Report di classificazione:")
            print(class_report)

            # Visualizza la matrice di confusione come heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()

    def denormalize_v1(self, img):
        return np.transpose((img / 2 + 0.5).numpy(), (1, 2, 0))


    def show_preview(self, trainloader):

        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        cols = 5
        rows = math.ceil(len(images) / cols)

        _, axs = plt.subplots(rows, cols, figsize=(10, 3))
        axs = axs.reshape(rows * cols)
        for ax, im, lb in zip(axs, images, labels):
            ax.imshow(self.denormalize_v1(im))
            ax.set_title(self.classes[lb.item()])
            ax.grid(False)
        plt.show()