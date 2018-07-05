import cv2 as cv
from pca import PCA
import pandas as pd

img = cv.imread('random_img/win_small.jpeg', cv.IMREAD_GRAYSCALE)
a_ratio = img.shape[0]/img.shape[1] 
df = pd.DataFrame(img.copy(), index=None, columns=None)
p = PCA(4)
p.loadData(df)
img_PCA = p.transform()
pixels = img_PCA.shape[0]*img_PCA.shape[1]
compressed_img = img_PCA.reshape()
print(img_PCA[0].shape)
print(img_PCA[1])
# reconstruct = p.reconstruct(img_PCA[0]).astype(int)
# print(reconstruct.shape)
cv.imwrite('random_img/win_small_original.jpeg', img)
cv.imwrite('random_img/win_small_compressed.jpeg', reconstruct) 