import os
from PIL import Image
import numpy as np
import cv2
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

class CLAHETransform:
    """
    Custom transform to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance texture details like fat lines in salmon/trout.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: CLAHE applied image.
        """
        # Convert PIL image to OpenCV format (numpy array)
        img_np = np.array(img)
        
        # Check if image is RGB
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Convert RGB to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            
            # Split LAB channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel (Luminance)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            
            # Merge enhanced L-channel with original A and B channels
            limg = cv2.merge((cl, a, b))
            
            # Convert back to RGB
            final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            # Return as PIL Image
            return Image.fromarray(final_img)
        
        return img # Return original if not RGB

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
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), # Force texture learning
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)), # Enhance fat lines
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transforms for Validation/Test (Resize + Normalize only + CLAHE)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)), # Apply CLAHE to test set too
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create Datasets for each split
    train_dataset = SalmonTroutDataset(data_dir, transform=train_transforms, split='train')
    val_dataset = SalmonTroutDataset(data_dir, transform=test_transforms, split='val')
    test_dataset = SalmonTroutDataset(data_dir, transform=test_transforms, split='test')

    # Create DataLoaders
    # Using num_workers=0 to avoid multiprocessing issues in some environments, or keep 2 if it was working
    # Previous code had num_workers=2. I will keep it but maybe set to 0 if issues arise.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
