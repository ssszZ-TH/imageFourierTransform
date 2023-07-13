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

cv.imwrite("imgFourier.png",imgFourier)
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
imgImagingrary_invers = np.fft.ifftshift(imgFourier_inverse)

