import rasterio
import numpy as np

# Band paths (Red, Green, Blue)
# Sentinel-2: B04=Red, B03=Green, B02=Blue
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Band paths (Red, Green, Blue)
# Sentinel-2: B04=Red, B03=Green, B02=Blue
red_path = "B04.tif"
green_path = "B03.tif"
blue_path = "B02.tif"
output_path = "mosiac_rgb.tif"
preview_path = "preview_mosaic.png"

def normalize(band):
    # Sentinel-2 raw is approx 0-10000. 
    # Visual range usually capped around 3000-4000 (reflectance 0.3-0.4)
    # Clip to max 3500 for good brightness, then scale to 0-255
    band = band.astype(float)
    band = np.clip(band, 0, 3500)
    band = (band / 3500) * 255
    return band.astype(np.uint8)

print("Reading bands...")
try:
    with rasterio.open(red_path) as src_r:
        red = src_r.read(1)
        profile = src_r.profile

    with rasterio.open(green_path) as src_g:
        green = src_g.read(1)

    with rasterio.open(blue_path) as src_b:
        blue = src_b.read(1)

    print("Normalizing and stacking (UInt16 -> UInt8)...")
    red_n = normalize(red)
    green_n = normalize(green)
    blue_n = normalize(blue)

    # Free memory
    del red, green, blue

    # Update profile for 3 channels, UInt8
    profile.update(
        count=3,
        driver='GTiff',
        dtype='uint8'
    )

    print(f"Writing RGB image to {output_path}...")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(red_n, 1)
        dst.write(green_n, 2)
        dst.write(blue_n, 3)
        
        # Create a downsampled preview for the user
        print(f"Saving preview to {preview_path}...")
        # Decimate by factor of 10 for speed/size
        r_small = red_n[::10, ::10]
        g_small = green_n[::10, ::10]
        b_small = blue_n[::10, ::10]
        rgb_small = np.dstack((r_small, g_small, b_small))
        
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_small)
        plt.title("Mosaic Preview (RGB)")
        plt.axis('off')
        plt.savefig(preview_path)
        plt.close()

    print("Done.")

except FileNotFoundError as e:
    print(f"Error: Missing band file: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
