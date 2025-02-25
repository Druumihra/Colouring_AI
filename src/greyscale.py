from PIL import Image;
import os
import pathlib

data_dir = pathlib.Path("./")

def check_images(image_name):
    og_count = len(list(data_dir.glob('images/original/*')))
    gs_count = len(list(data_dir.glob('images/greyscaled/*')))
    if og_count != gs_count: 
        return pathlib.Path(f"{data_dir}/greyscaled/greyscaled-{image_name}").exists()
 

def grey_scale_images():
    if not pathlib.Path("./images/greyscaled").exists():
        os.mkdir("./images/greyscaled")
    for image in data_dir.glob('Images/original/*') :
        image_extension_check = image.name.split(".")
        length = len(image_extension_check)
        if image_extension_check[length-1] == "avif" :
            return
        if check_images(image.name):
            im = Image.open(image)
            im = im.convert("L")
            image_name = image.name.split(".")
            im.save(fp=f"{data_dir}/Images/greyscaled/{image_name[0]}-greyscaled.{image_name[1]}")
