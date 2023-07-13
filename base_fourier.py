import numpy as np
import cv2 as cv

img = cv.imread("desktop.png",cv.IMREAD_GRAYSCALE)

#datatype เป็น float 32 bit
img = img.astype(np.float32)

#take fourier
imgF = np.fft.fft2(img)

#shift x y  0 0 ไปที่ตรงกลางเพื่อให้เห็นได้ง่ายขึ้น
imgFourier = np.fft.fftshift(imgF)

# magnitute and phase
imgReal = np.real(imgF)
imgImagingrary = np.imag(imgF)
imgMagnitute = np.sqrt(imgReal**2 + imgImagingrary**2)
imgPhysical = np.arctan2(imgImagingrary , imgReal)


cv.imwrite("imgReal.png",imgReal)
cv.imwrite("imgIMagingary.png",imgImagingrary)
cv.imwrite("imgMagnitute.png",imgMagnitute)
cv.imwrite("imgPhysical.png",imgPhysical)

### invers กลับ เป็นรูปเดิม
## invers fourier transformation
imgReal_invers = imgMagnitute * np.cos(imgPhysical)
imgImagingrary_invers = imgMagnitute*np.sin(imgPhysical)

imgFourier_inverse =  imgReal_invers + imgImagingrary_invers*1j

imgFourier_inverse = np.fft.ifftshift(imgFourier_inverse)
img_invers = np.fft.ifftshift(imgFourier_inverse)

imgImagingrary_invers = np.real(img_invers)
img_invers = img_invers.astype(np.uint8)
cv.imwrite("input.png",img)
cv.imwrite("output.png",img_invers)

## displya magnitude
imgMagnitute = np.log(1+imgMagnitute)
imgMagnitute = cv.normalize(imgMagnitute, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
cv.imwrite("magnitude_output.png",imgMagnitute)