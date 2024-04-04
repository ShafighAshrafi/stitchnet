from torch.utils.data import DataLoader


def generate_dataloader(dataset, batch_size=64, shuffle=True):
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)