from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open('pick.png') #here should be path
model = LatexOCR()
print(model(img))