import datasets
import os
import numpy as np
from PIL import Image

def read_uint_image(filepath, height, width, dtype=np.uint16):
    with open(filepath, 'rb') as f:
        img_data = np.fromfile(f, dtype=dtype)
    img_data = img_data.reshape((height, width))
    img_data = np.transpose(img_data, (1,0))
    img_data = img_data.astype(np.float32) / 4095.0
    return img_data

def read_float_image(filepath, height, width, dtype=np.float32):
    with open(filepath, 'rb') as f:
        img_data = np.fromfile(f, dtype=dtype)
    img_data = img_data.reshape((height, width))
    img_data = np.transpose(img_data, (1,0))
    img_data = img_data/ 4095.0
    return img_data


class PairedImageDataset(datasets.GeneratorBasedBuilder):
    '''
    This script deals with images stored at one single folder.
    '''
    VERSION = datasets.Version("2.0.0")
    
    def __init__(self, img_height=512, img_width=512, splits=None, *args, **kwargs):
        self.img_height = img_height
        self.img_width = img_width
        
        # Option 1: by ratio. For patients, cases must be seprated
        # self.splits = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
        # Option 2: by index.
        # self.splits = {'train': (0, 800), 'validation': (800, 900), 'test': (900, 1000)}
        self.splits = splits if splits is not None else {datasets.Split.TRAIN: 0.8, datasets.Split.VALIDATION: 0.1, datasets.Split.TEST: 0.1} 
        
        super().__init__(*args, **kwargs)
    
    def _info(self):
        return datasets.DatasetInfo(
            description = 'Paired Dataset Loader', 
            features=datasets.Features({
                'imageA': datasets.Array2D(dtype='float32', shape=(self.img_height, self.img_width)),
                'imageB': datasets.Array2D(dtype='float32', shape=(self.img_height, self.img_width)),
            }),
        )

    def _split_generators(self, dl_manager):
        local_path = self.config.data_dir
        
        if isinstance(self.splits[datasets.Split.TRAIN], int): # Ratio-based selection     
            all_images = [img for img in sorted(os.listdir(os.path.join(local_path, 'imageB'))) if img.endswith('.raw')]
            total_images = len(all_images)
            split_indices = {}
            start_idx = 0
            for split_name, split_ratio in self.splits.items():
                end_idx = start_idx + int(total_images * split_ratio)
                split_indices[split_name] = (start_idx, end_idx)
                start_idx = end_idx
            
        else:
            split_indices = self.splits
        
        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={
                    'imageA_dir': os.path.join(local_path, 'imageL30'),
                    'imageB_dir': os.path.join(local_path, 'imageB'),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                },
            ) for split_name, (start_idx, end_idx) in split_indices.items()
        ]

    def _generate_examples(self, imageA_dir, imageB_dir, start_idx, end_idx):    	
        all_images = [img for img in sorted(os.listdir(imageA_dir)) if img.endswith('.raw')]
        selected_images = all_images[start_idx:end_idx]
        
        for img_filename in selected_images:
            if img_filename.endswith('.raw'):
                imgA_filepath = os.path.join(imageA_dir, img_filename)             
                imgB_filepath = os.path.join(imageB_dir, img_filename)
                imgA_data = read_float_image(imgA_filepath, self.img_height, self.img_width)
                imgB_data = read_uint_image(imgB_filepath, self.img_height, self.img_width)               
                
                yield img_filename, {'imageA': imgA_data , 'imageB': imgB_data}
                
