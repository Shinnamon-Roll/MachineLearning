import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (string): 'train', 'val', 'test', or 'all'.
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
        
        # Define the exact folder names for each split
        # Note the case sensitivity: Salmon Train/Test/valid vs Trout train/test/valid
        
        folders_to_load = []
        
        if split == 'all':
            folders_to_load = [
                (os.path.join(salmon_base, 'Salmon Train'), 0),
                (os.path.join(salmon_base, 'Salmon valid'), 0),
                (os.path.join(salmon_base, 'Salmon Test'), 0),
                (os.path.join(trout_base, 'Trout train'), 1),
                (os.path.join(trout_base, 'Trout valid'), 1),
                (os.path.join(trout_base, 'Trout test'), 1),
            ]
        elif split == 'train':
            folders_to_load = [
                (os.path.join(salmon_base, 'Salmon Train'), 0),
                (os.path.join(trout_base, 'Trout train'), 1),
            ]
        elif split == 'val':
            folders_to_load = [
                (os.path.join(salmon_base, 'Salmon valid'), 0),
                (os.path.join(trout_base, 'Trout valid'), 1),
            ]
        elif split == 'test':
            folders_to_load = [
                (os.path.join(salmon_base, 'Salmon Test'), 0),
                (os.path.join(trout_base, 'Trout test'), 1),
            ]
            
        for folder_path, label in folders_to_load:
            self._add_images_from_folder(folder_path, label)
            
        print(f"Loaded {len(self.image_paths)} images for split: {split}")

    def _add_images_from_folder(self, folder_path, label):
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            # Try to be helpful with case sensitivity or common typos if needed
            # But for now, strict path following
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
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

def get_dataloaders(data_dir, batch_size=32, split_strategy=None):
    """
    Args:
        data_dir (string): Root directory of the dataset.
        batch_size (int): Batch size for DataLoader.
        split_strategy (tuple): IGNORED in this version as we use pre-split folders.
                                Kept for backward compatibility with existing calls.
    """
    
    # Data Augmentation for Training
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transforms for Validation/Test (Resize + Normalize only)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create Datasets for each split
    train_dataset = SalmonTroutDataset(data_dir, transform=train_transforms, split='train')
    val_dataset = SalmonTroutDataset(data_dir, transform=val_transforms, split='val')
    test_dataset = SalmonTroutDataset(data_dir, transform=val_transforms, split='test')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
