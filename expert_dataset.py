from torch.utils.data import Dataset


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root):
        self.data_root = data_root
        # Your code here

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        pass
