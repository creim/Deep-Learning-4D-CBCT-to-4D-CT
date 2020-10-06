'''Combining corresponding CBCT and CT images in one PNG File'''

from PIL import Image
import glob

cbct_images = []
for img in sorted(glob.glob('/home/creim/Desktop/pix2pix-tensorflow-official/datasets/Train-Thorax-7-ohne-Test-mit/train-CBCT/*.png')):
    cbct_images.append(Image.open(img))

ct_images = []
for img in sorted(glob.glob('/home/creim/Desktop/pix2pix-tensorflow-official/datasets/Train-Thorax-7-ohne-Test-mit/train-CT/*.png')):
    ct_images.append(Image.open(img))

iterator = 0
for img1, img2 in zip(cbct_images, ct_images):
    
    new_img = Image.new('L', (img1.width + img2.width, img1.height))    
    new_img.paste(img1, (0,0))
    new_img.paste(img2, (img1.width, 0))
    new_img.save('/home/creim/Desktop/pix2pix-tensorflow-official/datasets/Train-Thorax-7-ohne-Test-mit/train/' + str(iterator) +'.png')
    #new_img.save('/home/creim/Desktop/pix2pix-tensorflow-official/datasets/Train-Thorax-Varian-Neu-ohne-Test-mit/train/Slice-Bilder-50.png')
    iterator += 1