from diffusers import UNet2DModel, UNet2DConditionModel 

class UNet2DModel_Inpainting(UNet2DModel):
    def __init__(self, sample_size, in_channels, out_channels):
        super().__init__(
			sample_size=sample_size,  # the target image resolution
			in_channels=in_channels,  # the number of input channels, 3 for RGB images
			out_channels=out_channels,  # the number of output channels
			layers_per_block=2,  # how many ResNet layers to use per UNet block
			block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
			down_block_types=( 
				"DownBlock2D",  # a regular ResNet downsampling block
				"DownBlock2D", 
				"DownBlock2D", 
				"DownBlock2D", 
				"AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
				"DownBlock2D",
			), 
			up_block_types=(
				"UpBlock2D",  # a regular ResNet upsampling block
				"AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
				"UpBlock2D", 
				"UpBlock2D", 
				"UpBlock2D", 
				"UpBlock2D"  
			),
		)
        
    def forward(
        self,
        sample,
        timestep,
        class_labels = None,
        return_dict = True,
    	):
        x = super().forward(sample, timestep, class_labels, return_dict)
        return x


class UNet2DModel_Superres(UNet2DModel):
    def __init__(self, sample_size, in_channels, out_channels):
        super().__init__(
			sample_size=sample_size,  # the target image resolution
			in_channels=in_channels,  # the number of input channels, 3 for RGB images
			out_channels=out_channels,  # the number of output channels
			layers_per_block=2,  # how many ResNet layers to use per UNet block
			block_out_channels=(160, 320, 640, 1280),  # the number of output channes for each UNet block
			down_block_types=( 
				"DownBlock2D", 
				"DownBlock2D", 
				"DownBlock2D", 
				"AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
			), 
			up_block_types=(
				"AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
				"UpBlock2D", 
				"UpBlock2D", 
				"UpBlock2D", 
			),
		)
        
    def forward(
        self,
        sample,
        timestep,
        class_labels = None,
        return_dict = True,
    	):
        x = super().forward(sample, timestep, class_labels, return_dict)
        return x
		
		
		
		
