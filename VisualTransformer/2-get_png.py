import argparse
import os
import h5py
os.add_dll_directory("C:\\ProgramData\\Anaconda3\\envs\\AI11\\Library\\openslide-win64-20231011\\bin")
import openslide

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str, default='',
					help='你存放病理图像的文件夹')
parser.add_argument('--h5_source', type=str, default='', help='creat_patches.py文件的save_dir')
parser.add_argument('--patch_size', type = int, default=256, help='切片的大小 需要与creat_patches.py文件保持一致')
parser.add_argument('--save_dir', type = str, default='./ceshi', help='patch的保存路径')
parser.add_argument('--patch_level', type=int, default=0, help='切片的level 需要与creat_patches.py文件保持一致')
parser.add_argument('--type', type=str, default='tif', help='需要操作的文件格式 tif或者svs')

if __name__ == '__main__':
    # Set your arguments here
    args = {
        'source': '.\\wsi_file\\',
        'h5_source': '.\\h5_file\\',
        'patch_size': 512,##可以是256
        'save_dir': '.\\png_file\\',
        'patch_level': 0,##默认是0，就是最大倍数下切割，1就是倒数第二大的放大倍数
        'type': 'tif'##或者填为tif
    }

    print("Arguments:")
    for key, value in args.items():
        print(f"{key}: {value}")

    soruce_path = os.path.join(args['h5_source'], 'patches')
    slide_lis = os.listdir(soruce_path)
    for slide in slide_lis:
        slide_id = slide.split('.h5')[0]
        wsi = openslide.OpenSlide(os.path.join(args['source'], slide_id + '.' + args['type']))
        save_path = os.path.join(args['save_dir'], slide_id)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with h5py.File(os.path.join(soruce_path, slide), 'r') as f:
            dataset = f['coords']
            coords = dataset[:]
        for (x,y) in coords:
            image = wsi.read_region((x,y), args['patch_level'], (args['patch_size'], args['patch_size'])).convert('RGB')
            image.save(os.path.join(save_path, f'{x}_{y}.png'))

        

