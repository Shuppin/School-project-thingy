import os

from PIL import Image

class ImageHandler:
    """Functions for loading and generating alpha masks from image files"""
    
    def __init__(self):
        pass

    def __get_filename_from_path(self, path: str, ext: bool = False) -> str:
        """Gets the filename name from a whole path

        Arguments:
        - `path` (str): The path to extract the filename from
        - `ext` (Optional, bool): Return filename with/without extension

        Returns
        - `filename` (str)
        """
        if ext:
            return path.split("\\")[-1]
        else:
            return os.path.splitext(path.split("\\")[-1])[0]

    def load_images(self, path: str = ".\\img") -> list[Image.Image]:
        """Loads image files into Pillow image objects
        
        Arguments:
        - `path` (Optional, str): Path to directory containing files

        Raises:
        - `ValueError`
            - When a given filename ends with '_alpha'
        
        Returns:
        - `images` (list[PIL.Image.Image])
        """

        suit_image_objects: list[Image.Image] = []

        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".png"):
                    if self.__get_filename_from_path(file).endswith("_alpha"):
                        raise ValueError(f"Filename {repr(file)} cannot end with '_alpha'")

                    image = Image.open(os.path.join(root, file))
                    suit_image_objects.append(image)

        return suit_image_objects

    def generate_alpha_masks(self, images: list[Image.Image], threshold: int = 30) -> list[Image.Image]:
        """Genrates alpha masks for transparent images
        
        Arguments:
        - `images` (list[PIL.Image.Image]): List of images to generate alpha masks for
        - `threshold` (Optional, int) = 30: The threshold at which a pixel is determined transparent or opaque

        Returns:
        - `masks` (list[PIL.Image.Image]): List of generated alpha masks
        """

        alpha_masks: list[Image.Image] = []

        for image in images:

            alpha_mask = Image.new('RGBA', (image.width, image.height))

            for x in range(image.width):
                for y in range(image.height):
                    _,_,_,a = image.getpixel((x,y))
                    if a < threshold:
                        alpha_mask.putpixel((x,y), (0,0,0,0))
                    else:
                        alpha_mask.putpixel((x,y), (255,255,255,255))

            alpha_masks.append(alpha_mask)

        return alpha_masks
        
        
