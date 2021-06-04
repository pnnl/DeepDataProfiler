import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from torchvision import datasets, models, transforms

from deep_data_profiler.models import SimpleCNN

import copy
import time

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    """
    Trains a pytorch model with the given data, criterion, and optimizer
    
    Parameters
    ----------
    model: nn.Module
        The model to be trained
    dataloaders: dict
        A dict containing DataLoaders for the training and validation sets,
        dataloaders['val'] holds the DataLoader for the validation set and
        dataloaders['train'] holds the DataLoader for the training set.
    criterion: nn.Module
        A loss function from nn.modules.loss that will be used to train the model
    optimizer: optim.Optimizer
        An optimizer that will be used to train the model
    num_epochs: int, optional, default=25
        The number of epochs to train the model for
    is_inception: bool, optional, default=False
        True if the model is Inception v3
        From source: "Inception v3 [...] architecture uses an auxiliary
        output and the overall model loss respects both the auxiliary output 
        and the final output."
        
    Returns
    -------
    model: nn.Module
        The trained model
    val_acc: list
        A list containing the validation accuracy at each epoch of training
    
    Notes
    -----
    From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    """
    Notes
    -----
    From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True):
    """
    Notes
    -----
    From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    Modified to reflect models currently supported by DDP
    """
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def train_mnist(valid_ratio=0.9, num_epochs=10):
    """
    Trains a simple CNN model on the MNIST dataset
    
    Parameters
    ----------
    valid_ratio: float, optional, default=0.9
        A decimal representation of the training/validation split
        (by default, 90% of the original training set will be used to
        train the model, the other 10% will be used for validation.)
    num_epochs: int, optional, default=10
        The number of epochs to train the model for (the simple CNN 
        model achieves high accuracy with only 10 epochs of training)
    
    Returns
    -------
    model: nn.Module
        The trained simple CNN model
    val_acc: list
        A list containing the validation accuracy at each epoch of training
        
    Notes
    -----
    Modified from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    # load model
    model = SimpleCNN()
    
    # define transforms to resize and normalize data
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(3),
            transforms.RandomRotation(5, fill=(0,)),
            transforms.RandomCrop(28, padding = 2),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.1307, 0.1307, 0.1307], std = [0.3081, 0.3081, 0.3081])
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean = [mean], std = [std])
        ])
    }
    
    # load data with defined transforms
    ROOT = '.data'
    image_datasets = {'train': datasets.MNIST(root = ROOT,
                                              train = True, 
                                              download = True, 
                                              transform = train_transforms)}
    
    # split train data into train data and validation data
    n_train_examples = int(len(image_datasets['train']) * valid_ratio)
    n_valid_examples = len(image_datasets['train']) - n_train_examples
    image_datasets['train'], image_datasets['val'] = data.random_split(image_datasets['train'],
                                                                       [n_train_examples, n_valid_examples])   
    # overwrite validation train transforms with test transforms
    image_datasets['val'] = copy.deepcopy(image_datasets['val'])
    image_datasets['val'].dataset.transform = data_transforms['val']
    
    # Create training and validation dataloaders
    dataloaders_dict = {x : torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=4) for x in ['train', 'val']}
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model = model.to(device)    

    # Set up optimizer and loss fxn
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Train and evaluate
    model_ft, hist = train_model(model, dataloaders_dict, criterion,
                                 optimizer, num_epochs=num_epochs)
    return model_ft, hist

def train_cifar10(model_name, valid_ratio=0.9, batch_size=64, num_epochs=25, feature_extract=True):
    """
    Starting from either a VGG16 or Resnet18 model pretrained on ImageNet,
    retrains the classifier for the CIFAR-10 dataset
    
    Parameters
    ----------
    model_name: string
        The name of the model architecture to be used,
        must be either 'vgg' or 'resnet'
    valid_ratio: float, optional, default=0.9
        A decimal representation of the training/validation split
        (by default, 90% of the original training set will be used to
        train the model, the other 10% will be used for validation.)
    lr: float, optional, default=7e-5
        The learning rate for retraining layers in the classifier
        (the learning rate for feature layers will be lr / 10)
    
    Returns
    -------
    model: nn.Module
        The retrained VGG16 or Resnet18 model
    val_acc: list
        A list containing the validation accuracy at each epoch of training
    
    Notes
    -----
    Modified from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    # load model
    model_ft, input_size = initialize_model(model_name, 10, feature_extract=feature_extract)
    
    # define transforms to resize and normalize data
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # load data with defined transforms
    ROOT = '.data'
    image_datasets = {'train': datasets.CIFAR10(ROOT, 
                                                train = True, 
                                                download = True, 
                                                transform = data_transforms['train'])}
    # split train data into train data and validation data
    n_train_examples = int(len(image_datasets['train']) * valid_ratio)
    n_valid_examples = len(image_datasets['train']) - n_train_examples
    image_datasets['train'], image_datasets['val'] = data.random_split(image_datasets['train'],
                                                                       [n_train_examples, n_valid_examples])   
    # overwrite validation train transforms with test transforms
    image_datasets['val'] = copy.deepcopy(image_datasets['val'])
    image_datasets['val'].dataset.transform = data_transforms['val']    
    
    # Create training and validation dataloaders
    dataloaders_dict = {x : torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=4) for x in ['train', 'val']}
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss() 
    
   # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
                                 optimizer_ft, num_epochs=num_epochs,
                                 is_inception=(model_name=="inception"))
    
    return model_ft, hist
    
