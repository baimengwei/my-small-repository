#error if not exist
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import  QApplication,QMainWindow
from PyQt5 import QtGui, QtCore
from PyQt5 import QtCore, QtWidgets
from PyQt5.Qt import QFont
import neuralNetwork as nNPackage
import myUIActivation
from PyQt5.Qt import QFont
from PyQt5 import Qt
#nN.checkParameter()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(830, 700)

            palette1 = QtGui.QPalette()
            palette1.setColor(palette1.Background,QtGui.QColor(255,255,255))
            MainWindow.setPalette(palette1)
            MainWindow.setWindowFlags(Qt.Qt.FramelessWindowHint|Qt.Qt.WindowStaysOnTopHint)
            MainWindow.setAttribute(Qt.Qt.WA_TranslucentBackground)


            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            
            #train button
            self.pushButton = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton.setGeometry(QtCore.QRect(30, 40, 110, 25))
            self.pushButton.setObjectName("pushButton")
            #static text 
            self.label_1 = QtWidgets.QLabel(self.centralwidget)
            self.label_1.setGeometry(QtCore.QRect(160, 40, 50, 25))
            #train parameter
            self.trainEpochs = QtWidgets.QLineEdit(self.centralwidget)
            self.trainEpochs.setGeometry(QtCore.QRect(210, 40, 20, 25))
            self.trainEpochs.setObjectName("trainEpochs")
            #train label 
            self.label = QtWidgets.QLabel(self.centralwidget)
            self.label.setGeometry(QtCore.QRect(240, 40, 51, 21))
            self.label.setObjectName("label")
        
            self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
            self.progressBar_2.setGeometry(QtCore.QRect(360, 40, 118, 25))
            self.progressBar_2.setProperty("value", 0)
            self.progressBar_2.setObjectName("progressBar")
            
            #aWayToMakePCAData button
            self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_3.setGeometry(QtCore.QRect(30, 100, 110, 25))
            self.pushButton_3.setObjectName("pushButton_3")
            #aWayToMakePCAData label
            self.label_3 = QtWidgets.QLabel(self.centralwidget)
            self.label_3.setGeometry(QtCore.QRect(240, 105, 54, 12))
            self.label_3.setObjectName("label_3")
            
            #aWayToMakePCAData my own button
            self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_4.setGeometry(QtCore.QRect(30, 160, 110, 25))
            self.pushButton_4.setObjectName("pushButton_4")
            #aWayToMakePCAData my own label
            self.label_4 = QtWidgets.QLabel(self.centralwidget)
            self.label_4.setGeometry(QtCore.QRect(240, 165, 54, 12))
            self.label_4.setObjectName("label_4")
            #draw a line between train/aWayToMakePCAData and build a neural netwrok
            self.line = QtWidgets.QFrame(self.centralwidget)
            self.line.setGeometry(QtCore.QRect(530, 10, 10, 1000))
            self.line.setFrameShape(QtWidgets.QFrame.VLine)
            self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.line.setObjectName("line")
            
            
            #show error array in label
            self.label_8 = QtWidgets.QLabel(self.centralwidget)
            self.label_8.setGeometry(QtCore.QRect(60, 220, 54, 12))
            self.label_8.setObjectName("label_8")            
            
            #capture button
            self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_6.setGeometry(QtCore.QRect(30, 440, 110, 25))
            self.pushButton_6.setObjectName("pushButton_6")
            #capture label
            self.label_5 = QtWidgets.QLabel(self.centralwidget)
            self.label_5.setGeometry(QtCore.QRect(160, 445, 71, 20))
            self.label_5.setObjectName("label_5")
            #save button
            self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_8.setGeometry(QtCore.QRect(445, 440, 60, 25))
            self.pushButton_8.setObjectName("pushButton_8")            

            #recognition button
            self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_7.setGeometry(QtCore.QRect(30, 500, 110, 25))
            self.pushButton_7.setObjectName("pushButton_7")
            #static text named: param for change to binary picture
            self.label_9 = QtWidgets.QLabel(self.centralwidget)
            self.label_9.setGeometry(QtCore.QRect(160, 500, 50, 25))
            #train parameter
            self.threshold = QtWidgets.QLineEdit(self.centralwidget)
            self.threshold.setGeometry(QtCore.QRect(210, 500, 30, 25))
            self.threshold.setObjectName("threshold")  
            #static text named:block for delete small block in the picture
            self.label_10 = QtWidgets.QLabel(self.centralwidget)
            self.label_10.setGeometry(QtCore.QRect(250, 500, 50, 25))
            #train parameter
            self.blockSize = QtWidgets.QLineEdit(self.centralwidget)
            self.blockSize.setGeometry(QtCore.QRect(300, 500, 30, 25))
            self.blockSize.setObjectName("blockSize")  
            #capture picture
            self.label_6 = QtWidgets.QLabel(self)
            self.label_6.setGeometry(30, 550, 90, 90)     

            #restart nerualNetwork
            self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_1.setGeometry(QtCore.QRect(550, 40, 71, 25))
            self.pushButton_1.setObjectName("pushButton_1")
            #static text label : input.hidden.outputs.learning rate
            self.label_11 = QtWidgets.QLabel(self.centralwidget)
            self.label_11.setGeometry(QtCore.QRect(550, 90, 50, 25))                        
            #input nodes
            self.inputNodes = QtWidgets.QLineEdit(self.centralwidget)
            self.inputNodes.setGeometry(QtCore.QRect(550, 110, 30, 25))
            self.inputNodes.setObjectName("inputNodes")
            #hidden nodes
            self.hiddenNodes = QtWidgets.QLineEdit(self.centralwidget)
            self.hiddenNodes.setGeometry(QtCore.QRect(590, 110, 30, 25))
            self.hiddenNodes.setObjectName("hiddenNodes")
            #output nodes
            self.outputNodes = QtWidgets.QLineEdit(self.centralwidget)
            self.outputNodes.setGeometry(QtCore.QRect(635, 110, 30, 25))
            self.outputNodes.setObjectName("outputNodes")
            #learning rate
            self.learningRate = QtWidgets.QLineEdit(self.centralwidget)
            self.learningRate.setGeometry(QtCore.QRect(680, 110, 30, 25))
            self.learningRate.setObjectName("learningRate")
            #static for random mode
            self.label_12 = QtWidgets.QLabel(self.centralwidget)
            self.label_12.setGeometry(QtCore.QRect(550, 145, 50, 25))                
            #random mode
            self.rand = QtWidgets.QRadioButton(self.centralwidget)
            self.rand.setGeometry(QtCore.QRect(550, 160, 80, 20))
            self.rand.setObjectName("rand")                        
            self.normal = QtWidgets.QRadioButton(self.centralwidget)
            self.normal.setGeometry(QtCore.QRect(635, 160, 80, 20))
            self.normal.setObjectName("normal")            
            self.randomMode = QtWidgets.QButtonGroup(self)
            self.randomMode.addButton(self.rand, 1)
            self.randomMode.addButton(self.normal,2)
            #static for random seed
            self.label_13 = QtWidgets.QLabel(self.centralwidget)
            self.label_13.setGeometry(QtCore.QRect(550, 190, 50, 25))             
            #random seed 
            self.WihSeed = QtWidgets.QLineEdit(self.centralwidget)
            self.WihSeed.setGeometry(QtCore.QRect(590, 210, 30, 25))
            self.WihSeed.setObjectName("WihSeed")
            self.WhoSeed = QtWidgets.QLineEdit(self.centralwidget)
            self.WhoSeed.setGeometry(QtCore.QRect(635, 210, 30, 25))
            self.WhoSeed.setObjectName("WhoSeed")    
            #static for activate function
            self.label_14 = QtWidgets.QLabel(self.centralwidget)
            self.label_14.setGeometry(QtCore.QRect(550, 245, 50, 25))                                  
            #activate function selection
            self.sigmoid = QtWidgets.QRadioButton(self.centralwidget)
            self.sigmoid.setGeometry(QtCore.QRect(550, 270, 80, 20))
            self.sigmoid.setObjectName("sigmoid")
            self.softsign = QtWidgets.QRadioButton(self.centralwidget)
            self.softsign.setGeometry(QtCore.QRect(550, 290, 80, 20))
            self.softsign.setObjectName("softsign")
            self.relu = QtWidgets.QRadioButton(self.centralwidget)
            self.relu.setGeometry(QtCore.QRect(550, 310, 80, 20))
            self.relu.setObjectName("relu")
            self.tanh = QtWidgets.QRadioButton(self.centralwidget)
            self.tanh.setGeometry(QtCore.QRect(635, 270, 80, 20))
            self.tanh.setObjectName("tanh")
            self.softplus = QtWidgets.QRadioButton(self.centralwidget)
            self.softplus.setGeometry(QtCore.QRect(635, 290, 80, 20))
            self.softplus.setObjectName("softplus")                        
            #radioButton collection
            self.activateFunction = QtWidgets.QButtonGroup(self)
            self.activateFunction.addButton(self.sigmoid, 1)
            self.activateFunction.addButton(self.softsign, 2)
            self.activateFunction.addButton(self.relu, 3)
            self.activateFunction.addButton(self.tanh, 4)
            self.activateFunction.addButton(self.softplus, 5) 
            
            #No. of hidden layer static text 
            self.label_15 = QtWidgets.QLabel(self.centralwidget)
            self.label_15.setGeometry(QtCore.QRect(550, 345, 50, 25))
            #text of No. of hidden layer
            self.noOfHiddenLayer = QtWidgets.QLineEdit(self.centralwidget)
            self.noOfHiddenLayer.setGeometry(QtCore.QRect(700, 338, 30, 25))
            self.noOfHiddenLayer.setObjectName("noOfHiddenLayer")     
            
            MainWindow.setCentralWidget(self.centralwidget)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
            self.menubar.setObjectName("menubar")
            MainWindow.setMenuBar(self.menubar)
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)

            
            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)
            pass
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "手写数字识别测试评估系统"))
        
        #MainWindow.mousePressEvent(self,self.onActivatedShowPoint)
        
        self.pushButton.setText(_translate("MainWindow", "train"))
        self.pushButton.setFont(QFont("Roman times",10,QFont.Bold))
        self.pushButton.clicked.connect(self.onActivatedTrain)
        self.label.setText(_translate("MainWindow", "note"))
        self.label_1.setText(_translate("MainWindow", "epochs:"))
        self.trainEpochs.setText(_translate("MainWindow", "1"))
        
        self.pushButton_3.setText(_translate("MainWindow", "test"))
        self.pushButton_3.clicked.connect(self.onActivatedTest)
        self.pushButton_3.setFont(QFont("Roman times",10,QFont.Bold))
        self.label_3.setText(_translate("MainWindow", "note"))
        self.label_8.setText(_translate("MainWindow", "note"))
        
        self.pushButton_4.setText(_translate("MainWindow", "test my own"))
        self.pushButton_4.clicked.connect(self.onActivatedTestMyOwn)
        self.pushButton_4.setFont(QFont("Roman times",10,QFont.Bold))
        
        self.label_4.setText(_translate("MainWindow", "note"))
        
        self.pushButton_6.setText(_translate("MainWindow", "capture"))
        self.pushButton_6.clicked.connect(self.onActivatedCamera)
        self.pushButton_6.setFont(QFont("Roman times",10,QFont.Bold))
        self.label_5.setText(_translate("MainWindow", "answer"))
        self.pushButton_8.setText(_translate("MainWindow", "save"))
        self.pushButton_8.clicked.connect(self.onActivatedSaveWritting)
        self.pushButton_8.setFont(QFont("Roman times",10,QFont.Bold))
        
        self.pushButton_7.setText(_translate("MainWindow", "recognition"))
        self.pushButton_7.clicked.connect(self.onActivatedAnswer)
        self.pushButton_7.setFont(QFont("Roman times",10,QFont.Bold))
        self.label_9.setText(_translate("MainWindow", "param: "))
        self.threshold.setText(_translate("MainWindow", "0.7"))
        self.label_10.setText(_translate("MainWindow", "block: "))
        self.blockSize.setText(_translate("MainWindow", "18"))
        
        self.pushButton_1.setText(_translate("MainWindow", "forget"))
        self.pushButton_1.clicked.connect(self.onActivatedRestart)
        self.pushButton_1.setFont(QFont("Roman times",10,QFont.Bold))
        self.label_11.setText(_translate("MainWindow", "input.hidden.outputs.learning rate"))
        self.label_11.adjustSize()
        self.inputNodes.setText(_translate("MainWindow", "784"))
        self.inputNodes.setEnabled(True)
        self.hiddenNodes.setText(_translate("MainWindow", "100"))
        self.outputNodes.setText(_translate("MainWindow", "10"))
        self.outputNodes.setEnabled(False)
        self.learningRate.setText(_translate("MainWindow", "0.1"))
        self.label_12.setText(_translate("MainWindow", "select random mode:"))
        self.label_12.adjustSize()
        self.rand.setText(_translate("MainWindow", "rand"))
        self.normal.setText(_translate("MainWindow", "normal"))
        self.rand.setChecked(True)
        #random seed
        self.label_13.setText(_translate("MainWindow", "seed:  Wih    Who"))
        self.label_13.adjustSize()        
        self.WihSeed.setText(_translate("MainWindow", "4"))
        self.WhoSeed.setText(_translate("MainWindow", "5"))
        self.label_14.setText(_translate("MainWindow", "select the activate function:"))
        self.label_14.adjustSize()                
        self.sigmoid.setText(_translate("MainWindow", "sigmoid"))
        self.sigmoid.setChecked(True)
        self.softsign.setText(_translate("MainWindow", "softsign"))
        self.relu.setText(_translate("MainWindow", "relu"))
        self.tanh.setText(_translate("MainWindow", "tanh"))
        self.softplus.setText(_translate("MainWindow", "softplus"))
        
        self.label_15.setText(_translate("MainWindow", "number of hidden layer:"))
        self.label_15.adjustSize()
        self.noOfHiddenLayer.setText(_translate("MainWindow", "1"))
        



        palette1 = QtGui.QPalette()
        palette1.setColor(palette1.Background,QtGui.QColor(255,255,255))


        #restart neural network
        self.onActivatedRestart()

    def onActivatedTrain(self):
        myUIActivation.onActivatedTrain(self)
        pass

    
    def onActivatedTest(self):
        myUIActivation.onActivatedTest(self)
        pass
    def onActivatedTestMyOwn(self):
        myUIActivation.onActivatedTestMyOwn(self)
        pass   
    def onActivatedCamera(self):
        myUIActivation.onActivatedCamera(self)
        pass 
    def onActivatedAnswer(self):
        myUIActivation.onActivatedAnswer(self)
        pass

    def mouseMoveEvent(self, event):
        self.statusBar().showMessage(str(event.pos().x())+':'+str(event.pos().y()))
        pass
    def onActivatedRestart(self):
        myUIActivation.onActivatedRestart(self)
        pass
    def onActivatedSaveWritting(self):
        myUIActivation.onActivatedAnswer(self,1)
        pass
    
    
    
    
    