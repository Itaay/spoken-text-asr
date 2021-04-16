import torch


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
print('device is: {}'.format(dev))


def allocate_iterable(items_to_allocate):
    return [item.to(device) for item in items_to_allocate]