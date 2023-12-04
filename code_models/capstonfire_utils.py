import os
import json
import cv2

import tqdm.notebook
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms
from torchviz import make_dot
import torchmetrics
import torchmetrics.classification


class FireDataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        self.classes = sorted(os.listdir(root_dir))
        # Store the folder names in a dictionary as the class names alongside the class numeric label
        self.class_to_idx = {} 
        for i, cls in enumerate(self.classes):
            self.class_to_idx[cls] = i

        self.data = self._load_data()

    def _load_data(self):
        data = []
        for class_name in self.classes: # Fetch the folders through the class name
            class_path = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_path): # Fetch the images inside each folders
                img_path = os.path.join(class_path, filename) # Obtain the name of the current image
                data.append((img_path, self.class_to_idx[class_name])) # Add the image to a list paired with its class' numeric label
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # data[idx]
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        img = self.transforms(img)

        return img, label


def split_dataset_into_dataloaders(dataset, batch_size, train_size, val_size, test_size):

    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[train_size, val_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model: torch.nn.Module, criterion, optimizer, base_path: str, model_name: str, num_epochs: int, train_loader: DataLoader, val_loader: DataLoader):
    pretrained_models_path = os.path.join(base_path, "trained_models")
    hist_file_path = os.path.join(pretrained_models_path, model_name + '.json')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    model = model.to(device)

    history = {
        "avg_train_loss_list": [],
        "train_accuracy_list": [],
        "avg_val_loss_list": [],
        "val_accuracy_list": []
    }

    skip_saving : bool = False
    if model_name + '.pt' in os.listdir(pretrained_models_path):
        skip_saving = True
        model.load_state_dict(torch.load(os.path.join(pretrained_models_path, model_name +'.pt')))
        
        with open(hist_file_path, 'r') as json_file:
            history = json.load(json_file)

    # if model was never trained, enter loop
    else:
        for epoch in tqdm.notebook.tqdm(list(range(num_epochs))):
            model.train()

            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # zero the parameter gradients
                outputs = model(inputs)  # forward
                loss = criterion(outputs, labels)  # calculate loss
                loss.backward()  # backward
                optimizer.step()  # optimize

                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = correct_train / total_train
                
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                correct_val = 0
                total_val = 0
                val_loss = 0.0

                for inputs_val, labels_val in val_loader:
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val)

                    _, predicted_val = torch.max(outputs_val, 1)
                    total_val += labels_val.size(0)
                    correct_val += (predicted_val == labels_val).sum().item()

                    val_loss += loss_val.item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct_val / total_val

            history["avg_train_loss_list"].append(avg_train_loss)
            history["train_accuracy_list"].append(train_accuracy)
            history["avg_val_loss_list"].append(avg_val_loss)
            history["val_accuracy_list"].append(val_accuracy)

            print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        print('Finished Training')

        if not skip_saving:
        # save pytorch model
            torch.save(model.state_dict(), os.path.join(pretrained_models_path, model_name + '.pt'))

        # Write history dictionary to a JSON file
        with open(hist_file_path, 'w+') as json_file:
            json.dump(history, json_file)

    return history


def plot_loss(first_axis, second_axis, _is_accuracy = False):
    fig = plt.figure()
    plt.plot(first_axis, color='teal', label='train_accuracy' if _is_accuracy else 'train_loss')
    plt.plot(second_axis, color='orange', label='val_accuracy' if _is_accuracy else 'val_loss')
    fig.suptitle('Accuracy' if _is_accuracy else 'Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


def plot_accuracy(first_axis, second_axis):
    plot_loss(first_axis, second_axis, _is_accuracy=True)


def calculate_metrics(model, test_loader):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    model = model.to(device)

    recall = torchmetrics.classification.Recall(task='multiclass',num_classes=2).to(device)
    precision = torchmetrics.classification.Precision(task='multiclass',num_classes=2).to(device)
    f1 = torchmetrics.classification.F1Score(task='multiclass',num_classes=2).to(device)

    with torch.no_grad():
        for data in tqdm.notebook.tqdm(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # forward
            #predicted_binary = (torch.sigmoid(outputs) > 0.5).float()
            predictions = torch.argmax(outputs, dim=1)

            recall.update(predictions, labels)
            precision.update(predictions, labels)
            f1.update(predictions, labels)

    print('Recall on the test set: %.2f' % (recall.compute()))
    print('Precision on the test set: %.2f' % (precision.compute()))
    print('F1 Score on the test set: %.2f' % (f1.compute()))

    
def run_video(filename, model : torch.nn.Module, transforms: torchvision.transforms):
    """
        Inputs: 
        - filename: the path to the video to run
        - model: the CNN model to make the prediction on each frame of the video
        - transforms: the transform to be applied to each frame of the video in order to make it
          compatible with the model
        Outputs: 
        - Arr[is_fire_percentage, is_not_fire_percentage]
    """

    predictions = [0, 0]
    total_frames = 1

    cap = cv2.VideoCapture(filename)

    while True:
        ret, image = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break
        
        # Make a prediction for the current frame if a model was given
        if model != None and transforms != None:
            draw = image.copy()
            draw = cv2.resize(draw, (640, 480))
            draw = transforms(draw)

            outputs = model(draw)
            #prob = torch.sigmoid(outputs)
            _, prob = torch.max(outputs, 1)

            if prob > 0.5:
                color = (0, 255, 0)
                predictions[1] += 1
            else:
                color = (0, 0, 255)
                predictions[0] += 1
            
            cv2.putText(image, "fire" if prob.item() == 0 else "non_fire", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display the frame
        cv2.imshow('framename', image)

        total_frames += 1

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return {'fire': predictions[0]/total_frames, 'non_fire': predictions[1]/total_frames}