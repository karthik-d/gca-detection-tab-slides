from torchvision import transforms 

from ..config import config

#TODO: Write merger to combine transforms pipelined at multiple locations

name_transform_map = dict(
    img_to_tensor = transforms.ToTensor(),
    resize_to_input_shape = transforms.Resize(
        size=config.get('INPUT_XY'),
        interpolation=transforms.InterpolationMode.BILINEAR
    )
)

def get_transforms(pre_transforms):
    # Applying only basic transforms for now

    transform_instances = list(map(
        name_transform_map.get, 
        config.get('TRANSFORMS')
    ))

    return transforms.Compose(
        transforms=transform_instances
    )