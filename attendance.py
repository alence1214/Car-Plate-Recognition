import sys
import cv2
import imutils
import numpy as np
import pytesseract
from anpr import PyImageSearchANPR
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication
from PyQt5.QtCore import pyqtSlot, QTimer, QDate, Qt, QTime, QDateTime
from attendance_widget import Ui_Widget

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

class Attendance_Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.image = None
        self.capture = None
        self.camera_name = 0
        self.timer = QTimer(self)
        self.auto_stop_timer = QTimer(self)
        self.time_count = 0

        self.date_today = QDate.currentDate()
        self.ui.label_date.setText(self.date_today.toString())
        self.attendance_name_set = set()

        self.start_time = None
        self.end_time = None

        self.detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
        self.anpr = PyImageSearchANPR(debug=True)

        self.post_url = 'https://prod.mspeducare.com/mobapi'
        self.headers = {
            'Content-Type': 'application/json'
        }

        self.ui.btn_start.clicked.connect(self.startCapture)
        self.ui.btn_end.clicked.connect(self.stopCapture)
        self.ui.btn_return.clicked.connect(self.btn_return_clicked)

    @pyqtSlot()
    def btn_return_clicked(self):
        self.capture = None
        self.image = None
        self.timer.stop()
        self.timer = None
        self.auto_stop_timer.stop()
        self.auto_stop_timer = None
        self.close()

    @pyqtSlot()
    def startCapture(self):
        self.start_time = QTime.currentTime()
        self.ui.label_start_time.setText("Start Time: " + self.start_time.toString())
        self.camera_name = 0
        if cv2.VideoCapture("1.mp4"):
            self.capture = cv2.VideoCapture("1.mp4")
        else:
            self.capture = None
            QMessageBox.warning(self, 'Camera Error!', f'Camera Connection error occured!')
            with open('error.log', 'w') as error_log:
                error_log.write(f'{QDateTime.currentDateTime().toString()}: Camera Connection error occured: Please Connect Camera!\n')
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(40)
        self.auto_stop_timer.timeout.connect(self.auto_stop)
        self.auto_stop_timer.start(60000)

    @pyqtSlot()
    def auto_stop(self):
        if self.time_count >= 60:
            self.stopCapture()
            self.time_count = 0
        self.time_count += 1
        print(self.time_count)
    
    @pyqtSlot()
    def stopCapture(self):
        self.capture.release()
        self.end_time = QTime.currentTime()
        self.ui.label_end_time.setText("End Time: " + self.end_time.toString())
        self.timer.stop()
        self.ui.video_screen.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.auto_stop_timer.stop()
        self.time_count = 0
        self.attendance_name_set.clear()
        
    def update_frame(self):
        if self.capture != None:
            _, self.image = self.capture.read()
            self.display_image(self.image)
        
    def display_image(self, image):
        image = self.detect_plate_haar(image)

        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        self.ui.video_screen.setPixmap(QPixmap.fromImage(outImage))
        self.ui.video_screen.setScaledContents(True)
        
    def detect_plate_anpr(self, image):
        image = imutils.resize(image, width=640)
        
        (lpText, lpCnt) = self.anpr.find_and_ocr(image, psm=9, clearBoarder=True)
        
        if lpText is not None and lpCnt is not None:
            box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
            box = box.astype("int")
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
            
            (x, y, w, h) = cv2.boundingRect(lpCnt)
            lpText = "".join([c if ord(c) < 128 else "" for c in lpText]).strip()
            cv2.putText(image, lpText, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
        return image
    
    def detect_plate_contour(self, image):
        # image = imutils.resize(image, width=720)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

        edged = cv2.Canny(gray_image, 10, 200)

        contours, new = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = []
        idx = 7

        for c in contours:
            # approximate the license plate contour
            contour_perimeter = cv2.arcLength(c, False)
            approx = cv2.approxPolyDP(c, 0.01 * contour_perimeter, True)
            # print(approx)
            cv2.drawContours(image, [approx], -1 , (0, 255, 0), 2)
            # Look for contours with 4 corners
            if len(approx) == 4:
                
                screenCnt.append(approx)

                # find the coordinates of the license plate contour
                x, y, w, h = cv2.boundingRect(c)
                new_img = image [ y: y + h, x: x + w]

                # stores the new image
                cv2.imwrite('./'+str(idx)+'.png',new_img)
                idx += 1
                break
        for ap in screenCnt:
            x, y, w, h = cv2.boundingRect(ap)
            plate = image[y:y+h, x:x+w]
            kernel = np.ones((1,1),np.uint8)
            plate = cv2.dilate(plate, kernel, iterations =1)
            plate = cv2.erode(plate, kernel, iterations =1)
            plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
            (thresh,plate) = cv2.threshold(plate_gray,127,255, cv2.THRESH_BINARY)

            plate_num = pytesseract.image_to_string(plate, lang='eng')
            plate_num = ''.join(e for e in plate_num if e.isalnum())
            if len(plate_num) >= 4:
                cv2.rectangle(image,(x,y),(x+w,y+h),(51,51,255),2)
                cv2.rectangle(image,(x,y-40),(x+w,y),(51,51,255),-1)
                cv2.putText(image,plate_num,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8,color = (255, 255, 255),thickness = 2)
        return image

    def detect_plate_haar(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plates = self.detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in plates:
            print(x, y, w, h)
            a,b = (int(0.02*image.shape[0]),int(0.018*image.shape[1]))
            plate = image[y+a:y+h-a, x+b:x+w-b, :]
            # image processing
            kernel = np.ones((1,1),np.uint8)
            plate = cv2.dilate(plate, kernel, iterations =1)
            plate = cv2.erode(plate, kernel, iterations =1)
            plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
            (thresh,plate) = cv2.threshold(plate_gray,127,255, cv2.THRESH_BINARY)

            plate_num = pytesseract.image_to_string(plate, lang='eng')
            plate_num = ''.join(e for e in plate_num if e.isalnum())


            print(plate_num)
            cv2.rectangle(image,(x,y),(x+w,y+h),(51,51,255),2)
            cv2.rectangle(image,(x,y-40),(x+w,y),(51,51,255),-1)
            cv2.putText(image,plate_num,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8,color = (255, 255, 255),thickness = 2)
        
        return image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Attendance_Widget()
    ui.show()
    sys.exit(app.exec_())