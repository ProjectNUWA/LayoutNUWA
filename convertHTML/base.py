import torch
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
import torch_geometric  
from torch_geometric.data import DataLoader  
  
class BaseDataset(InMemoryDataset):
    labels = []
    _label2index = None
    _index2label = None
    _colors = None

    def __init__(self, path, split, transform):
        assert split in ['train', 'val', 'test']
        self.path = path
        super().__init__(self.path, transform)
        
        idx = self.processed_file_names.index('{}.pt'.format(split))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def label2index(self):
        if self._label2index is None:
            self._label2index = dict()
            for idx, label in enumerate(self.labels):
                self._label2index[label] = idx
        return self._label2index

    @property
    def index2label(self):
        if self._index2label is None:
            self._index2label = dict()
            for idx, label in enumerate(self.labels):
                self._index2label[idx] = label
        return self._index2label

    @property
    def colors(self):
        if self._colors is None:
            n_colors = self.num_classes
            colors = sns.color_palette('husl', n_colors=n_colors)
            self._colors = [tuple(map(lambda x: int(x * 255), c))
                            for c in colors]
        return self._colors

    @property
    def raw_file_names(self):
        raw_dir = Path(self.path)
        if not raw_dir.exists():
            return []
        return [p.name for p in raw_dir.iterdir()]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        raise NotImplementedError
    
    
