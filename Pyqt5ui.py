from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi

from PyQt5.QtWidgets import *

import sys
import cv2
import numpy as np
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        loadUi('demo.ui', self)
        self.setWindowIcon(QtGui.QIcon("vet.png"))

        self.image = None
        self.actionOpen.triggered.connect(self.open_img)
        self.actionSave.triggered.connect(self.save_img)
        self.actionPrint.triggered.connect(self.createPrintDialog)
        self.actionQuit.triggered.connect(self.QuestionMessage)
        self.actionBig.triggered.connect(self.big_Img)
        self.actionSmall.triggered.connect(self.small_Img)
        self.actionQt.triggered.connect(self.AboutMessage)
        self.actionAuthor.triggered.connect(self.AboutMessage2)


        self.actionTranslation.triggered.connect(self.translation)

        self.actionOtsu_Thresold.triggered.connect(self.otsu_threshold)
        self.actionGray_Scale.triggered.connect(self.gray_scale)
        self.actionBorder_Constant.triggered.connect(self.border_constant)
        self.actionBorder_Replicate.triggered.connect(self.border_replicate)
        self.actionGamma_2.triggered.connect(self.gamma)
        self.actionHistogram_Ciz.triggered.connect(self.histogram_ciz)
        self.actionHistogram_Esitle.triggered.connect(self.histogram_esitleme)
        self.actionGraidentL2.triggered.connect(self.l2_graident)
        self.actionDeriche_Fitlre.triggered.connect(self.deriche_filtre)
        self.actionHarris_Kose.triggered.connect(self.harris_kose)


        self.btn_otsu.clicked.connect(self.otsu_threshold)
        self.btn_gray.clicked.connect(self.gray_scale)
        self.btn_constant.clicked.connect(self.border_constant)
        self.btn_replicate.clicked.connect(self.border_replicate)
        self.btn_gamma.clicked.connect(self.gamma)
        self.btn_histoCiz.clicked.connect(self.histogram_ciz)
        self.btn_histoEsitle.clicked.connect(self.histogram_esitleme)
        self.btn_graident.clicked.connect(self.l2_graident)
        self.btn_deriche.clicked.connect(self.deriche_filtre)
        self.btn_harris.clicked.connect(self.harris_kose)
        self.btn_faceCascade.clicked.connect(self.Face_Cascade)
        self.btn_contourDetection.clicked.connect(self.Contour_Detection)
        self.btn_morpho.clicked.connect(self.Morfolojik)


        # Smoothing
        self.actionBlur.triggered.connect(self.blurred_filter)
        self.actionBox_Filter.triggered.connect(self.box_filter)
        self.actionMedian_Filter.triggered.connect(self.median_filter)
        self.actionBilateral_Filter.triggered.connect(self.bilateral_filter)
        self.actionGaussian_Filter.triggered.connect(self.gaussian_filter)
        self.actionFiltre_2D.triggered.connect(self.filter_2D)

        self.btn_blur.clicked.connect(self.blurred_filter)
        self.btn_boxF.clicked.connect(self.box_filter)
        self.btn_medianF.clicked.connect(self.median_filter)
        self.btn_bilateralF.clicked.connect(self.bilateral_filter)
        self.btn_gaussF.clicked.connect(self.gaussian_filter)
        self.btn_2dF.clicked.connect(self.filter_2D)

        # Slider
        self.gammaSlider.valueChanged.connect(self.gamma)
        self.gaussSlider.valueChanged.connect(self.gaussian_filter)
        self.constanstSlider.valueChanged.connect(self.border_constant)
        self.graidentMinSlider.valueChanged.connect(self.l2_graident)
        self.graidentMaxSlider.valueChanged.connect(self.l2_graident)
        self.contourSlider.valueChanged.connect(self.Contour_Detection)

        # Filter
        #self.actionMedian_threshold_2.triggered.connect(self.median_threshold)
        self.actionDirectional_Filtering_2.triggered.connect(self.directional_filtering)
        self.actionDirectional_Filtering_3.triggered.connect(self.directional_filtering2)
        self.actionDirectional_Filtering_4.triggered.connect(self.directional_filtering3)
        self.action_Butterworth.triggered.connect(self.butter_filter)
        self.action_Notch_filter.triggered.connect(self.notch_filter)

        #clear
        self.btn_clear.clicked.connect(lambda: self.clearImage(window=2))

    @pyqtSlot()
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.tmp = self.image
        self.displayImage()

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        # image.shape[0] là số pixel theo chiều Y
        # image.shape[1] là số pixel theo chiều X
        # image.shape[2] lưu số channel biểu thị mỗi pixel
        img = img.rgbSwapped() # chuyển đổi hiệu quả một ảnh RGB thành một ảnh BGR.
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)# căn chỉnh vị trí xuất hiện của hình trên lable
        if window == 2:
            self.imgLabel2.setPixmap(QPixmap.fromImage(img))
            self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def clearImage(self, window=1):
        if window == 1:
            self.imgLabel.clear()
        elif window == 2:
            self.imgLabel2.clear()

    def open_img(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\DELL\PycharmProjects\DemoPro', "Image Files (*)")
        if fname:
            self.loadImage(fname)
        else:
            print("Invalid Image")

    def save_img(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\', "Image Files (*.png)")
        if fname:
            cv2.imwrite(fname, self.image) # Lưu trữ ảnh
            print("Error")

    def createPrintDialog(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)

        if dialog.exec_() == QPrintDialog.Accepted:
            self.imgLabel2.print_(printer)

    def big_Img(self):
        self.image = cv2.resize(self.image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def small_Img(self):
        self.image = cv2.resize(self.image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def SIZE(self , c):
        self.image = self.tmp
        self.image = cv2.resize(self.image, None, fx=c, fy=c, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def reset(self):
        self.image = self.tmp
        self.displayImage(2)

    def AboutMessage(self):
        QMessageBox.about(self, "About Qt - Qt Designer",
            "Qt is a multiplatform C + + GUI toolkit created and maintained byTrolltech.It provides application developers with all the functionality needed to build applications with state-of-the-art graphical user interfaces.\n"
            "Qt is fully object-oriented, easily extensible, and allows true component programming.Read the Whitepaper for a comprehensive technical overview.\n\n"

            "Since its commercial introduction in early 1996, Qt has formed the basis of many thousands of successful applications worldwide.Qt is also the basis of the popular KDE Linux desktop environment, a standard component of all major Linux distributions.See our Customer Success Stories for some examples of commercial Qt development.\n\n"

            "Qt is supported on the following platforms:\n\n"

                "\tMS / Windows - - 95, 98, NT\n"
                "\t4.0, ME, 2000, and XP\n"
                "\tUnix / X11 - - Linux, Sun\n"
                "\tSolaris, HP - UX, Compaq Tru64 UNIX, IBM AIX, SGI IRIX and a wide range of others\n"
                "\tMacintosh - - Mac OS X\n"
                "\tEmbedded - - Linux platforms with framebuffer support.\n\n"
                          
            "Qt is released in different editions:\n\n"
            
                "\tThe Qt Enterprise Edition and the Qt Professional Edition provide for commercial software development.They permit traditional commercial software distribution and include free upgrades and Technical Support.For the latest prices, see the Trolltech web site, Pricing and Availability page, or contact sales @ trolltech.com.The Enterprise Edition offers additional modules compared to the Professional Edition.\n\n"
                "\tThe Qt Open Source Edition is available for Unix / X11, Macintosh and Embedded Linux.The Open Source Edition is for the development of Free and Open Source software only.It is provided free of charge under the terms of both the Q Public License and the GNU General Public License."
        )
    def AboutMessage2(self):
        QMessageBox.about(self, "About Author", "Người hướng dẫn:   NGÔ QUỐC VIỆT \n\n" 
                                                "Người thực hiện:\n" 
                                                    "\tPhan Hoàng Việt - 42.01.104.189"
                          )

    def QuestionMessage(self):
        message = QMessageBox.question(self, "Çıkış", "Uygulamadan çıkış yapılsınmı?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            print("Yes")
            self.close()
        else:
            print("No")

    def translation(self):
        self.image = self.tmp
        num_rows, num_cols = self.image.shape[:2]

        translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
        img_translation = cv2.warpAffine(self.image, translation_matrix, (num_cols, num_rows))
        self.image = img_translation
        self.displayImage(2)

    def erode(self , iter):
        self.image = self.tmp
        if iter > 0 :
            kernel = np.ones((4, 7), np.uint8)
            self.image = cv2.erode(self.tmp, kernel, iterations=iter)
        else :
            kernel = np.ones((2, 6), np.uint8)
            self.image = cv2.dilate(self.image, kernel, iterations=iter*-1)
        self.displayImage(2)

    def Canny(self):
        self.image = self.tmp
        if self.canny.isChecked():
            can = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.Canny(can, self.canny_min.value(), self.canny_max.value())
        self.displayImage(2)


#####################################Smoothing##########################################################################
    def blurred_filter(self):
        self.image = self.tmp
        self.image = cv2.blur(self.image, (5, 5))
        self.displayImage(2)
    def box_filter(self):
        self.image = self.tmp
        self.image = cv2.boxFilter(self.image, -1,(5,5))
        self.displayImage(2)
    def median_filter(self):
        self.image = self.tmp
        self.image = cv2.medianBlur(self.image,5)
        self.displayImage(2)
    def bilateral_filter(self):
        self.image = self.tmp
        self.image = cv2.bilateralFilter(self.image,9,75,75)
        self.displayImage(2)
    def gaussian_filter(self):
        self.image = self.tmp
        self.image = cv2.GaussianBlur(self.image,(5,5),0)
        self.displayImage(2)
    def filter_2D(self):    
        kernel=np.array([[-1,-1,-1],[-1,8,-1]])
        #kernel1=np.float32([[-1,-1,-1],[-1,8,-1],[-1.-1,-1]])
        sonuc=cv2.filter2D(self.image,-1,kernel)
        self.displayImage(2)

########################################Filter##########################################################################

    def otsu_threshold(self):
        self.image = self.tmp
        grayscaled = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.medianBlur(self.image,5)
        retval, otsu_threshold = cv2.threshold(grayscaled,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.image = otsu_threshold
        self.displayImage(2)

    def directional_filtering(self):
        self.image = self.tmp
        kernel = np.ones((3, 3), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)

    def directional_filtering2(self):
        self.image = self.tmp
        kernel = np.ones((5, 5), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)

    def directional_filtering3(self):
        self.image = self.tmp
        kernel = np.ones((7, 7), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)

    def butter_filter(self):
        self.image = self.tmp
        img_float32 = np.float32(self.image)

        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.image = np.fft.fftshift(dft)

        self.image = 20 * np.log(cv2.magnitude(self.image[:, :, 0], self.image[:, :, 1]))
        self.displayImage(2)

    def notch_filter(self):
        self.image = self.tmp
        self.displayImage(2)

    def gray_scale(self):
        self.image = self.tmp
        imageGray = cv2.cvtColor(self.image, int(cv2.COLOR_BGR2GRAY))
        #_re, threshold= cv2.threshold(imageGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.image = imageGray
        self.displayImage(2)
        
    def border_constant(self):
        #BORDER_CONSTANT:yeni eklenen tüm piksellerin değeri sabit bir değer yapılabilir
        border_color=(0, 0, 0)
        border_width = 10
        image_with_border_constant = cv2.copyMakeBorder(self.image, border_width, border_width, border_width, border_width, borderType=cv2.BORDER_CONSTANT, value=[120, 12, 240])
        self.image = image_with_border_constant
        self.displayImage(2)

    def border_replicate(self):
        #BORDER_REPLİCATE:yeni eklenen tüm piksellerin değeri resmin dışına en yakın piksel değeri ile belirlenir.
        border_color = (0, 0, 0)
        border_width =10
        image_with_border_replicate =cv2.copyMakeBorder(self.image, border_width,border_width,border_width,border_width,borderType=cv2.BORDER_REPLICATE,value=border_color)
        self.image = image_with_border_replicate
        self.displayImage(2)

    def gamma(self):
        def apply_gamma_correction(gamma=1.0):
            image_normalized = self.image / 255.0
            gamma_corrected=np.power(image_normalized, gamma)
            gamma_corrected=np.uint8(gamma_corrected* 255)
            return gamma_corrected
        gamma_value = 0.5
        gamma_corrected_image =apply_gamma_correction(gamma=gamma_value)
        self.image = gamma_corrected_image
        self.displayImage(2)

    def histogram_ciz(self):
        image1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #histogram hesapla
        hist = cv2.calcHist([image1], channels=[0], mask=None, histSize=[250], ranges=[0, 256])
        #Histogram eğrisini çiz
        plt.plot(hist)
        plt.title('Histogram Eğrisi')
        plt.xlabel('pixel değeri')
        plt.ylabel('pixel sayısı')
        plt.show()
     
    def histogram_esitleme(self):
        #Histogram eşitleme uygula
        resim = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        equalized_image=cv2.equalizeHist(resim)
        #orjinal eşitlemiş görüntüyü göster
        plt.subplot(1, 2, 1)
        plt.imshow(resim, cmap='gray')
        plt.title('Orjinal görüntü')
        plt.subplot(1, 2, 2)
        plt.imshow(resim, cmap='gray')
        plt.title('Histogram eşitleme sonrası')
        plt.show()

    def l2_graident(self):
        image=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        low_thresold=50
        high_thresold=150
        sonuc1=cv2.Canny(image,low_thresold,high_thresold,L2gradient=True)
        self.image = sonuc1
        self.displayImage(2)

        #f,eksen=plt.subplots(1,2,figsize=(17,7))
        #l2Graident=true gradyan hesabında ..
        #eksen[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),cmap="gray")
        #eksen[1].imshow(sonuc1,cmap="gray")
        #plt.show()
    
    def deriche_filtre(self):
        image=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #Deriche filtresi için kernel oluşturu
        alpha=0.5 #dehrice filtresi parametresi
        kernel_size=3 #Filtre Boyutu

        kx,ky=cv2.getDerivKernels(1, 1, kernel_size,normalize=True)
        dehriche_kernel_x=alpha*kx
        dehriche_kernel_y=alpha*ky

        #Görüntüyü Deriche filtresi ile türevle
        dehriche_x=cv2.filter2D(image,-1,dehriche_kernel_x)
        dehriche_y=cv2.filter2D(image,-1,dehriche_kernel_y)
        
        #Kenarları Birleştir
        edges=np.sqrt(dehriche_x**2 + dehriche_y**2)
        self.image = edges
        self.displayImage(2)
        #f,eksen=plt.subplots(1,2,figsize=(17,7))
        #eksen[0].imshow(image,cmap="gray")
        #eksen[1].imshow(edges,cmap="gray")
        #plt.show()
    
    def harris_kose(self):
        images=self.image
        gray=cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

        #Harrris köşe tespiti için parametreleri ayarlla
        corner_quality = 0.04
        min_distance = 10
        block_size = 3
        #Harris köşe tespiti uygula
        corners=cv2.cornerHarris(gray, block_size, 3, corner_quality)

        #Köşeleri belirili bir eşik değerine  göre seç
        corners = cv2.dilate(corners, None)
        images[corners > 0.01 *corners.max()]=[0, 0, 255]
        self.image = images
        self.displayImage(2)

    #HaarCascades----------
    def Face_Cascade(self):
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        conf_1 = self.image
        conf_2 = conf_1.copy()

        faces_1 = faceCascade.detectMultiScale(conf_1)
        for(x, y, w, h) in faces_1:
            cv2.rectangle(conf_1, (x, y), (x+w, y+h), (255, 0, 0), 10)
        faces_2 = faceCascade.detectMultiScale(conf_2, scaleFactor=1.3, minNeighbors=6)
        for (x, y, w, h) in faces_2:
            cv2.rectangle(conf_2, (x, y), (x + w, y + h), (255, 0, 0), 10)

        f, eksen = plt.subplots(1, 2, figsize=(20, 10))
        eksen[0].imshow(conf_1, cmap="gray")
        eksen[1].imshow(conf_2, cmap="gray")
        plt.show()

    def Contour_Detection(self):
        img = self.image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Kenarları tespit etmek için canny kenar dedektörünü kullan
        edges = cv2.Canny(gray, 50, 150)

        #Konturları bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Resmi üzerinde konturları çiz
        resim= cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        self.image = resim
        self.displayImage(2)

    #tamamla
    def Morfolojik(self):
        imgOrj = self.image
        imgBlur = cv2.medianBlur(imgOrj, 31)
        # Madeni para içindeki detayların sonucu etkilememesi için blurring yaptık
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        # Griye çevrilen resim THREH_BINARY_INV  ile arkaplan siyah,önplan beyaz yapılır
        ret, imgTH = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        # imhTH çıktısında ön plan nesneleri üzerinde kalan gürültüden kurtulmak için
        # morfolojik  operatörlerden openning islemi uygulanır
        imgOPN = cv2.morphologyEx(imgTH, cv2.MORPH_OPEN, kernel, iterations=7)

        # Arka plan alanı
        # dilate() fonksiyonu ie nesneler genişletilir ve kesin emin olduğumuz arkaplan
        # kısımları elde edilir
        sureBG = cv2.dilate(imgOPN, kernel, iterations=3)
        # ÖnPlan Alanı
        # distanceTransform() ile her pikselin en yakın sıfır değerine sahip piksele
        # olan mesafeleri hesaplanır.Nesnelerin merkez pikselleri yani sıfır piksellerine en
        # uzak nokta beyaz kalırken,siyah pikselere yaklaştıkça piksel değerleri düşer
        # böylece madeni para ,yani emin olduğumuz,ön plan pikselleri ortaya çıkması.....
        dist_transform = cv2.distanceTransform(imgOPN, cv2.DIST_L2, 5)

        # Eşikleme yap
        # Eşik değeri olarak hesaplanan maksimum mesafenin %70'den büyük olanlarının
        # piksel değeri 255 yapılarak sureFG elde edilmiştir
        ret, sureFG = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Bilinmeyen bölgeleri bul
        # Emin olduğumuz arkaplan ve ön plan arasında kalanın alanıdır.
        sureFG = np.uint8(sureFG)
        unknow = cv2.subtract(sureBG, sureFG)
        # Etiketleme islemi
        ret, markers = cv2.connectedComponents(sureFG, labels=5)
        # Bilinmeyen pikselleri etiketleme
        markers = markers + 1
        markers[unknow == 255] = 0
        # Watershed algoritması uygula
        markers = cv2.watershed(imgOrj, markers)
        contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        imgCopy = imgOrj.copy()
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(imgCopy, contours, i, (255, 0, 0), 5)

        f, eksen = plt.subplots(3, 3, figsize=(30, 30))
        eksen[0, 0].imshow(imgOrj)
        eksen[0, 1].imshow(imgBlur)
        eksen[0, 2].imshow(imgGray)
        eksen[1, 0].imshow(imgTH)
        eksen[1, 1].imshow(imgOPN)
        eksen[1, 2].imshow(sureBG)
        eksen[2, 0].imshow(dist_transform)
        eksen[2, 1].imshow(sureFG)
        eksen[2, 2].imshow(imgCopy)
        plt.show()



app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())

