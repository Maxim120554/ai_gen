from PIL import Image
import numpy as np

def remove_bg(input_path, output_path, thresh=190):
    img = Image.open(input_path).convert("RGBA")
    d = np.array(img).astype(int)
    r, g, b = d[:,:,0], d[:,:,1], d[:,:,2]
    light = (r > thresh) & (g > thresh) & (b > thresh)
    d[:,:,3] = np.where(light, 0, 255)
    Image.fromarray(d.astype(np.uint8)).save(output_path)

remove_bg("icons/building.png", "icons/building_transparent.png")