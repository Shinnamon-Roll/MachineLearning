import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y, path = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y, path
        
    def __len__(self):
        return len(self.subset)

class SalmonTroutDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train', 'test', or 'all'. 
                           'train' will combine 'Salmon Train', 'Salmon valid' and 'Trout train'.
                           'test' will combine 'Salmon Test' and 'Trout test'.
                           'all' will load all available images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Salmon', 'Trout']
        self.image_paths = []
        self.labels = []
        
        # Define folder mapping based on the provided structure
        # Salmon (Label 0)
        # Trout (Label 1)
        
        salmon_base = os.path.join(root_dir, 'Salmon!')
        trout_base = os.path.join(root_dir, 'Trout!')
        
        if mode == 'all':
            # Load everything
            self._add_images_from_folder(os.path.join(salmon_base, 'Salmon Train'), 0)
            self._add_images_from_folder(os.path.join(salmon_base, 'Salmon valid'), 0)
            self._add_images_from_folder(os.path.join(salmon_base, 'Salmon Test'), 0)
            self._add_images_from_folder(os.path.join(trout_base, 'Trout train'), 1)
            self._add_images_from_folder(os.path.join(trout_base, 'Trout test'), 1)
        elif mode == 'train':
            # Salmon Training Data
            self._add_images_from_folder(os.path.join(salmon_base, 'Salmon Train'), 0)
            self._add_images_from_folder(os.path.join(salmon_base, 'Salmon valid'), 0)
            
            # Trout Training Data
            self._add_images_from_folder(os.path.join(trout_base, 'Trout train'), 1)
            
        elif mode == 'test':
            # Salmon Testing Data
            self._add_images_from_folder(os.path.join(salmon_base, 'Salmon Test'), 0)
            
            # Trout Testing Data
            self._add_images_from_folder(os.path.join(trout_base, 'Trout test'), 1)
            
        print(f"Loaded {len(self.image_paths)} images for mode: {mode}")

    def _add_images_from_folder(self, folder_path, label):
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            return
            
        for filename in os.listdir(folder_path):
            if filename.startswith('.'):
                continue
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.image_paths.append(os.path.join(folder_path, filename))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or handle error appropriately
            # For simplicity, we might just skip or error out, but here let's try to return the next one
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

def get_dataloaders(data_dir, batch_size=32, split_strategy=(0.8, 0.1, 0.1)):
    # Data Augmentation for Training
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transforms for Testing (No augmentation)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full dataset first
    full_dataset = SalmonTroutDataset(data_dir, transform=None, mode='all')
    total_size = len(full_dataset)
    
    train_size = int(split_strategy[0] * total_size)
    val_size = int(split_strategy[1] * total_size)
    test_size = total_size - train_size - val_size
    
    import torch
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = SubsetWithTransform(train_subset, transform=train_transforms)
    val_dataset = SubsetWithTransform(val_subset, transform=test_transforms)
    test_dataset = SubsetWithTransform(test_subset, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
