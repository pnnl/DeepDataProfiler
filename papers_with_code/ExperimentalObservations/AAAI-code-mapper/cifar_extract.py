import os
import sys
import numpy as np
import h5py
import collections
from functools import partial
import torch
from tqdm import tqdm
import torchvision
from cifar_train import ResNet18
import torch.nn.functional as F
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_model(path, num_classes):
    model = ResNet18(num_classes)
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'])
    return model


def inv_normalize(data, mean, std):
    return torchvision.transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())(data)


def get_activations(model, dataset, restrict_to_label, is_sample=False, is_imgs=False):
    norm_mean = np.array((0.4914, 0.4822, 0.4465))
    norm_std = np.array((0.2023, 0.1994, 0.2010))
    # storage_dict keeps saving the activations as they come
    storage_dict = collections.defaultdict(list)

    def save_activation(name, mod, inp, out):
        storage_dict[name].append(out)

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    # Save all hooks to remove at the end
    hook_list = []
    for name, m in model.named_modules():
        # print(name, m)
        if type(m) == torch.nn.BatchNorm2d and not 'shortcut' in name:
            # partial to assign the layer name to each hook
            hook_list.append(m.register_forward_hook(partial(save_activation, name)))

    # forward pass through the full dataset
    icons = []
    img_list = []
    predictions = np.array([])
    with torch.no_grad():
        for batch, labels in tqdm(dataset, desc=f'Processing label={restrict_to_label}'):
            batch_sub = batch[labels == restrict_to_label]
            if is_imgs == True:
                img_list.append(batch_sub.numpy())
            out = model(batch_sub)
            _, prediction = out.max(1)
            prediction = prediction.numpy()
            predictions = np.concatenate((predictions, prediction))
            # out = model(batch[labels == restrict_to_label])
            icons += [np.expand_dims(inv_normalize(img, norm_mean, norm_std).permute(1, 2, 0), 0)
                      for img in batch[labels == restrict_to_label]]

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    if is_sample:
        storage_dict = {name: sample_batch_activation(torch.cat(outputs, 0)) for name, outputs in storage_dict.items()}
    else:
        storage_dict = {name: torch.cat(outputs, 0) for name, outputs in storage_dict.items()}
    storage_dict['icons'] = np.vstack(icons)
    storage_dict['predictions'] = predictions
    if is_imgs == True:
        storage_dict['imgs'] = np.vstack(img_list)

    # Remove all hooks
    [h.remove() for h in hook_list]
    return storage_dict


def sample_batch_activation(batch_activation):
    # Return one activation for layer while avoiding boundary patches
    batch_size, dim, patch_n, patch_m = batch_activation.shape
    assert patch_n == patch_m

    batch_indices = np.arange(batch_size)

    # Sample while ignoring boundary indices
    rand_indices = np.random.randint(1, patch_n, size=(batch_size, 2))

    return F.relu(batch_activation[batch_indices, :, rand_indices[:, 0], rand_indices[:, 1]]).numpy()

def read_activation(filepath, layer):
    with h5py.File(filepath, 'r') as f:
        activation = f[layer][:]
        return activation

if __name__ == '__main__':
    DATASET = sys.argv[1]

    norm_mean = np.array((0.4914, 0.4822, 0.4465))
    norm_std = np.array((0.2023, 0.1994, 0.2010))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(norm_mean.tolist(),
                                                                                  norm_std.tolist())])

    if DATASET == 'CIFAR10':
        train = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=False,
                                             transform=transforms)
        test = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=False,
                                            transform=transforms)
        num_classes = 10
    if DATASET == 'CIFAR100':
        train = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=True, download=False,
                                              transform=transforms)
        test = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=False, download=False,
                                             transform=transforms)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=2048, num_workers=4)
    testloader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=2048, num_workers=4)

    net = load_model(f'../logs/{DATASET}_ResNet18_Custom_Aug/checkpoints/best.pth', num_classes)
    net.eval()

    activation_dir = f'../activations/{DATASET.lower()}/resnet18_custom_aug/full_activations/'

    for label_filter in range(num_classes):
        activations = get_activations(net, trainloader, label_filter)
        with h5py.File(os.path.join(activation_dir, f'label{label_filter}.hdf5'), 'w') as out_file:
            [out_file.create_dataset(layer_name, data=layer_act) for layer_name, layer_act in
             activations.items()]
        del activations
