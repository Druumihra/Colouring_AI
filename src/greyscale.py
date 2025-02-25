from PIL import Image;
import pathlib

data_dir = pathlib.Path("./")

def check_images(image_name):
    og_count = len(list(data_dir.glob('images/original/*')))
    gs_count = len(list(data_dir.glob('images/greyscaled/*')))
    if og_count != gs_count: 
        return pathlib.Path(f"{data_dir}/greyscaled/greyscaled-{image_name}")
    else:
        return False
    

def grey_scale_images():
    for image in data_dir.glob('Images/original/*') :
        if check_images(image.name):
            im = Image.open(image)
            im = im.convert("L")
            image_name = image.name.split(".")
            im.save(fp=f"{data_dir}/Images/greyscaled/{image_name[0]}-greyscaled.{image_name[1]}")

grey_scale_images()