import cv2

def preprocess(image, gaussian_kernel):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.GaussianBlur(image, (gaussian_kernel, gaussian_kernel), 0)
	_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	return image
