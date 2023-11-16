# import the necessary packages
from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2
class PyImageSearchANPR:
	def __init__(self, minAR=4, maxAR=5, debug=False):
		# store the minimum and maximum rectangular aspect ratio
		# values along with whether or not we are in debug mode
		self.minAR = minAR
		self.maxAR = maxAR
		self.debug = debug

	def debug_imshow(self, title, image, waitKey=False):
		# check to see if we are in debug mode, and if so, show the
		# image with the supplied title
		if self.debug:
			cv2.imwrite(".\\output\\" + title + ".jpg", image)
			# check to see if we should wait for a keypress

	def locate_license_plate_candidates(self, gray, keep=5):
		# perform a blackhat morphological operation that will allow
		# us to reveal dark regions (i.e., text) on light backgrounds
		# (i.e., the license plate itself)
		rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
		blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
		self.debug_imshow("Blackhat", blackhat)

		squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
		light = cv2.threshold(light, 0, 255,
						cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		self.debug_imshow("Light Regions", light)

		gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
					dx=1, dy=0, ksize=1)
		gradX = np.absolute(gradX)
		(minVal, maxVal) = (np.min(gradX), np.max(gradX))
		gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
		gradX = gradX.astype("uint8")
		self.debug_imshow("Scharr", gradX)

		gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
		gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
		thresh = cv2.threshold(gradX, 0, 255,
						cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		self.debug_imshow("Grand Thresh", thresh)

		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)
		self.debug_imshow("Grad Erode/Dilate", thresh)

		thresh = cv2.bitwise_and(thresh, thresh, mask=light)
		thresh = cv2.dilate(thresh, None, iterations=2)
		thresh = cv2.erode(thresh, None, iterations=1)
		self.debug_imshow("Final", thresh, waitKey=True)

		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

		return cnts

	def locate_license_plate(self, gray, candidates, clearBorder=False):
		lpCnt = None
		roi = None

		for c in candidates:
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)

			if ar >= self.minAR and ar <= self.maxAR:
				lpCnt = c
				licensePlate = gray[y:y+h, x:x+w]
				roi = cv2.threshold(licensePlate, 0, 255,
							cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

				if clearBorder:
					roi = clear_border(roi)
				self.debug_imshow("License Plate", licensePlate)
				self.debug_imshow("ROI", roi, waitKey=True)
		
		return (roi, lpCnt)

	def build_tesseract_options(self, psm=7):
		alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
		options = "-c tessedit_char_whitelist={}".format(alphanumeric)

		options += "--pam {}".format(psm)
		return options

	def find_and_ocr(self, image, psm=7, clearBoarder=False):
		lpText = None

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		candidates = self.locate_license_plate_candidates(gray)
		(lp, lpCnt) = self.locate_license_plate(gray, candidates, clearBorder=clearBoarder)

		if lp is not None:
			options = self.build_tesseract_options(psm=psm)
			lpText = pytesseract.image_to_string(lp, config=options)
			self.debug_imshow("License Plate", lp)

		return (lpText, lpCnt)