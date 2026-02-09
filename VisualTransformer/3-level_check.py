import openslide

wsi_path = r'.\wsi_file\22-A-C-1HE_Wholeslide_Default_Extended.tif'
wsi = openslide.OpenSlide(wsi_path)

print(f"Number of levels: {wsi.level_count}")
print(f"Level dimensions: {wsi.level_dimensions}")
print(f"Level downsamples: {wsi.level_downsamples}")