# import the necessary packages
from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2

class PyImageSearchANPR:
        def __init__(self, minAR=3, maxAR=5, debug=False):
                #minimalni i maksimalni omjer visine i širine tablice
                #nalazimo li se u debug mode ili ne
                self.minAR = minAR
                self.maxAR = maxAR
                self.debug = debug
                
        def debug_imshow(self, title, image, waitKey=False):
        	#ako smo u debug mode, prikazati fotografiju nakon međukoraka
                if self.debug:
                        cv2.imshow(title, image)
                        if waitKey:
                                cv2.waitKey(0)
                                
        def locate_license_plate_candidates(self, gray, image, keep=5):
                rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 6))
                blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
                self.debug_imshow("Blackhat", blackhat)
                
                #isticanje svijetlih dijelova na slici
                squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
                light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                self.debug_imshow("Light Regions", light)
                
		#Scharrov gradijent, koji ističe rubove znakova na tablici, operacija
                #se obavlja na blackhat slici
                gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
                gradX = np.absolute(gradX)
                (minVal, maxVal) = (np.min(gradX), np.max(gradX))
                gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
                gradX = gradX.astype("uint8")
                self.debug_imshow("Scharr", gradX)
                
                #Gaussovo zamućivanje, closing operacija i binariziranje (threshold)
                gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
                gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
                thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                self.debug_imshow("Grad Thresh", thresh)
                
		#iteracije erozije i dilatacije u svrhu čišćenja slike od šuma
                thresh = cv2.erode(thresh, None, iterations=1)
                thresh = cv2.dilate(thresh, None, iterations=3)
                self.debug_imshow("Grad Erode/Dilate", thresh)
                
		#provlačenje slike kroz masku "light" u svrhu isticanja kandidata za
                #registarsku tablicu; još erozija i dilatacija
                thresh = cv2.bitwise_and(thresh, thresh, mask=light)
                thresh = cv2.dilate(thresh, None, iterations=2)
                thresh = cv2.erode(thresh, None, iterations=0)
                self.debug_imshow("Final", thresh, waitKey=True)
                
		#traženje kontura, sortiranje kontura od najveće do najmanje i zadržavanje
                #samo određenog broja kontura
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
		#vraća se lista kontura
                oriCopy = image.copy()
                for c in cnts:
                        cv2.drawContours(oriCopy, [c], -1, 255, 2)
                        self.debug_imshow("Contours", oriCopy)

                return cnts
	    
        def locate_license_plate(self, gray, candidates, clearBorder=False):
		#incijalizacija konture tablice i područja interesa tablice
                lpCnt = None
                roi = None
                #za sve kandidate 
                for c in candidates:
                        #crtanje pravokutnika oko konture i nalaženje omjera
                        #izmežu visine i širine tog pravokutnika
                        (x, y, w, h) = cv2.boundingRect(c)
                        ar = w / float(h)
			#kontura se uzima u obzir samo ako joj je omjer unutar zadanih vrijednosti
                        if ar >= self.minAR and ar <= self.maxAR:
				#sprema se kontura u lpCnt, slika tablice se prevodi u grayscale sliku
                                #i obavlja se binarizacija da bismo dobili područje interesa
                                lpCnt = c
                                licensePlate = gray[y:y + h, x:x + w]
                                roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
				#ako je uključena opcija clearBorder, očistiti sve piksele koji diraju rub slike
                                if clearBorder:
                                        roi = clear_border(roi)
				#prikazati tablicu i područje interesa
                                self.debug_imshow("License Plate", licensePlate)
                                self.debug_imshow("ROI", roi, waitKey=True)
                                break
		#vraća se područje interesa i kontura tablice
                return (roi, lpCnt)

        def build_tesseract_options(self, psm=7):
        	#postavljanje Tesseracta da čita slova i brojeve koji se mogu naći na tablici
                alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                options = "-c tessedit_char_whitelist={}".format(alphanumeric)
		#postavljanje načina (formata) čitanja (7 = linija teksta)
                options += " --psm {}".format(psm)
                #vraćanje opcija
                return options

        def find_and_ocr(self, image, psm=7, clearBorder=False):
                #incijalizacija teksta koji će se pročitati
                lpText = None
		#prebacivanje slike u grayscale, lociranje svih kandidata
		#i procesuiranje kandidata dok ne dođemo do tablice
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                candidates = self.locate_license_plate_candidates(gray, image)
                (lp, lpCnt) = self.locate_license_plate(gray, candidates, clearBorder=clearBorder)
		#obavlja se OCR samo ako je područje interesa pronađeno
                if lp is not None:
			#obavljanje OCR
                        options = self.build_tesseract_options(psm=psm)
                        lpText = pytesseract.image_to_string(lp, config=options)
                        self.debug_imshow("License Plate", lp)
		#vraća se pročitani tekst i pripadajuća kontura tablice
                return (lpText, lpCnt)	    

	    
