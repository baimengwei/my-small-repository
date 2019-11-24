from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
from skimage import io,transform,filters,morphology,exposure
import skimage.morphology as sm
import numpy as np
import numpy
from PyQt5.QtWidgets import  QFileDialog
from PyQt5.QtGui import QPixmap
import toolFunctions
import neuralNetwork as nNPackage
from skimage import measure,color
from prettytable import PrettyTable
from time import time
import cv2
import scipy.ndimage
#nN.checkParameter()

#输入标准图像，查询图像结果
def show_ans(self,img_dst):#图像转化与查询(255图像，0多)
    img_data = img_dst.reshape(784)
    img_data = (img_data /255.0 * 0.99 ) + 0.01
    op=self.nN.query(img_data)
    #print(op)
    #print(numpy.argmax(op))
    return numpy.argmax(op)

#show error data of digital in a table
#input arrayData information
#output: a draw
#others:predict :target for first and second in each pare of arrayData
def showArrayData(self,arrayData):
    table=[[0]*11 for i in range(10)]
    for i in range(int(len(arrayData)/2)):
        table[arrayData[i*2]][arrayData[i*2+1]+1] += 1
        pass
    x= PrettyTable(["p\\t",0,1,2,3,4,5,6,7,8,9])
    for i in range(10):
        table[i][0]=i
        x.add_row(table[i])
    self.label_8.setText(x.get_string())
    self.label_8.adjustSize()
    pass

def onActivatedTrain(self):
    self.label.setText("start training!")
    self.label.adjustSize()
    #MNIST train
    fname=QFileDialog.getOpenFileName(self, 'Open file', './')[0]
    #fname="mnist_train.csv"
    data_file = open(fname,'r')
    print(fname)
    data_list = data_file.readlines()
    data_file.close()
    self.step = 0
    count = 0
    activateFunctionSelect=self.activateFunction.checkedId()
    length=len(data_list)
    epochs = int(self.trainEpochs.text())
    startTime=time()   
    for e in range(epochs) :
        for record in data_list:
            all_values = record.split(',')
            #if select alter
            #inputs = toolFunctions.makeAlterMNIST(all_values)
            #inputs = inputs/255.0*0.99+0.01
            #MNIST org
            if(activateFunctionSelect==1):
                inputs = numpy.asfarray( all_values [1:])/255.0*0.99+0.01
                targets = numpy.zeros(self.nN.oNodes) + 0.01
                #print(all_values)
                targets[int (float(all_values[0]))] = 0.99
            elif(activateFunctionSelect==2):
                inputs = numpy.asfarray( all_values [1:])/255.0*0.5
                targets = numpy.zeros(self.nN.oNodes) -0.99
                targets[int (float(all_values[0]))] = 0.99
            elif(activateFunctionSelect==3):
                inputs = numpy.asfarray( all_values [1:])/255.0*0.99+0.01
                targets = numpy.zeros(self.nN.oNodes) + 0.01
                targets[int (float(all_values[0]))] = 10
            elif(activateFunctionSelect==4):
                inputs = numpy.asfarray( all_values [1:])/255.0*0.5
                targets = numpy.zeros(self.nN.oNodes) -0.99
                targets[int (float(all_values[0]))] = 0.99
            elif(activateFunctionSelect==5):
                inputs = numpy.asfarray( all_values [1:])/255.0*0.99+0.01
                targets = numpy.zeros(self.nN.oNodes) + 0.01
                targets[int (float(all_values[0]))] = 10
                pass
            self.nN.train(inputs,targets,activateFunctionSelect)
            #inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
            #self.nN.train(inputs_plusx_img.reshape(784),targets,activateFunctionSelect)
            #inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
            #self.nN.train(inputs_plusx_img.reshape(784),targets,activateFunctionSelect)            
            count+=1
            self.step = int(count/(length*epochs)*100)
            QApplication.processEvents()
            self.progressBar_2.setValue(self.step)
            fname
            pass
        pass
    self.nN.totalEpochs+=epochs
    duration =time()-startTime
    self.label.setText("complete!\nIt takes:"+str(round(duration,2))+"sec\n"+"total epochs:"+str(self.nN.totalEpochs))
    self.label.adjustSize()
    pass

def onActivatedTest(self):
    self.label_3.setText("start test MNIST")
    self.label_3.adjustSize() 
    fname=QFileDialog.getOpenFileName(self, 'Open file', './')[0]
    test_data_file = open(fname,'r')
    print(fname)
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scorecard = []
    prediction = []
    arrayData = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int (float(all_values[0]))
        inputs = numpy.asfarray( all_values [1:])/255.0*0.99+0.01
        outputs = self.nN.query(inputs)#for catch the target value
        label = numpy.argmax(outputs)
        prediction.append(label)
        if(label == correct_label):
            scorecard.append(1)
            #saveArray=numpy.asfarray(all_values)
            #saveArray=saveArray.astype(int)
            #toolFunctions.makeMNIST(saveArray,"saveForSelectTrain.csv")
        else:
            scorecard.append(0)
            pass
        arrayData.append(label)
        arrayData.append(correct_label)
        pass
    scorecard_array = numpy.asarray(scorecard)
    showArrayData(self,arrayData)
    #print ("performance = " ,str(scorecard_array.sum()),str(scorecard_array.size))
    self.label_3.setText("performance = " +str(round(scorecard_array.sum()/scorecard_array.size,4)))
    self.label_3.adjustSize() 
    pass

def onActivatedTestMyOwn(self):
    self.label_4.setText("start test my own")
    self.label_4.adjustSize()
    test__list=[]
    prediction_label =[]
    fault = 0
    fname = QFileDialog.getExistingDirectory(self, 'Open file', './')
    for j in range(5):
        for i in range(10):
            img_rgb = io.imread(fname+'/test'+str(j+1)+'_'+str(i)+'.png')
            img_gray = color.rgb2gray(img_rgb)
            img_data = toolFunctions.handlePicture(img_gray)
            predict = show_ans(self,img_data)
            prediction_label.append(predict)
            img_data = img_data.reshape(784)
            img_data = img_data.tolist()
            img_data.insert(0,i)
            if i != predict:
                fault+=1
                pass
            test__list.append(img_data)
            #toolFunctions.makeMNIST(img_data)
            pass
        pass
    toolFunctions.plotImageLabelsPrediction(test__list,prediction_label,labelEnable=1,option=1,num=50)
    self.label_4.setText("have shown on an window\n the socre is :"+str(1-fault/50))
    self.label_4.adjustSize()
            
def onActivatedCamera(self):
    self.label_5.setText("start capture and 'q' for capture and 'w' for exit ")
    self.label_5.adjustSize()
    print("opencv vision:",cv2.__version__)
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("digital capture",0);
    cv2.resizeWindow("digital capture", self.captureWidth, self.captureHeight);

    print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while(True):
        # 获取一帧
        ret, img = capture.read()
        cropped = img[0:self.captureHeight, 0:self.captureWidth]
        # 将这帧转换为灰度图
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("digital capture", cropped)
        c = cv2.waitKey(1)
        if  c == ord('w'):
            capture.release() #释放摄像头
            cv2.destroyAllWindows()#删除建立的全部窗口
            break
        if c == ord('q'):
            cv2.imwrite("youtemp1.png", cropped)
            capture.release() #释放摄像头
            cv2.destroyAllWindows()#删除建立的全部窗口
            self.label_6.setPixmap(QPixmap('youtemp1.png').scaled(self.captureWidth, self.captureHeight))
            self.label_6.adjustSize()
            break
    pass 

def onActivatedAnswer(self,save=0):

    self.label_5.setText("answers:")
    self.label_5.adjustSize()
    
    self.label_6.setPixmap(QPixmap('youtemp1.png').scaled(self.captureWidth, self.captureHeight))
    self.label_6.adjustSize()
    
    img_rgb = io.imread('youtemp1.png')
    plt.figure()
    plt.imshow(img_rgb)    
    img_gray = color.rgb2gray(img_rgb)
    #plt.figure()
    #plt.imshow(img_gray, cmap = 'Greys_r')   
    img_gray = toolFunctions.pictureBright(img_gray)
    #plt.figure()
    #plt.imshow(img_gray, cmap = 'Greys_r')          
    thd = float(self.threshold.text())
    point_l,point_r,point_t,point_b,labels=toolFunctions.pictureFindEdge(img_gray,multiple=1,threshold=thd)
    print(point_l);print(point_r);print(point_t);print(point_b)
    
    plt.figure()
    plt.imshow(labels)
    thd = int(self.blockSize.text())
    toolFunctions.deleteSmallPoint(point_l,point_r,point_t,point_b,threshold=thd)#去除部分可疑点，包括了过于小的点块和边界点块

    print(point_l);print(point_r);print(point_t);print(point_b)

    toolFunctions.sortPoint(point_l,point_r,point_t,point_b)#选择排序，按点左标签排序
    print(point_l);print(point_r);print(point_t);print(point_b)
          
    answerList = []
    imgList = []
    
    num= len(point_l)
    for i in range(0,num):
        image_array = img_gray[point_t[i]:point_b[i],point_l[i]:point_r[i]]
        #plt.figure()
        #plt.imshow(image_array,cmap='Greys_r')   
        image_array = exposure.rescale_intensity(image_array,in_range=(0,1),out_range=(0,255))
        #plt.figure()
        #plt.imshow(image_array,cmap='Greys_r')
        image_array = 255 - image_array  
        #plt.figure()
        #plt.imshow(image_array,cmap='Greys_r')
        #plt.figure()
        #plt.imshow(image_array,cmap='binary')                     
        image_array=toolFunctions.addEmpty(image_array,height_param=4,width_param=2)#增加合适的空白     
        #plt.figure()
        #plt.imshow(image_array,cmap='binary')    
        img_dst=transform.resize(image_array,(28,28),mode='constant') 
        #plt.figure()
        #plt.imshow(img_dst,cmap='binary')            
        img_dst=sm.erosion(img_dst,sm.square(1))#
        #plt.figure()
        #plt.imshow(img_dst,cmap='binary')
        #img_dst=sm.dilation(img_dst,sm.square(1))
        img_dst=filters.gaussian(img_dst,sigma=1)
        #plt.figure()
        #plt.imshow(img_dst,cmap='binary')
        img_dst=img_dst.astype(np.float32)
        img_dst = img_dst.reshape(784)
        imgList.append(img_dst)
        #print(img_dst.shape)
        #print()
        #ax.imshow(image_array,cmap='binary')
        answerList.append(show_ans(self,img_dst))
    toolFunctions.plotImageLabelsPrediction(imgList,answerList,labelEnable=0,option=1,num=num)
    if save==1:
        toolFunctions.storeWritting(imgList,"user.csv")
    self.label_5.setText("answer:"+str(answerList))
    self.label_5.adjustSize()
    pass
    
def onActivatedRestart(self):
    print("start a new neuralNetwork")
    inputNodes = int(self.inputNodes.text())
    outputNodes = int(self.outputNodes.text())
    hiddenNodes = int(self.hiddenNodes.text())
    learningRate = float(self.learningRate.text())
    activationFunction = self.activateFunction.checkedId()
    randomMode = self.randomMode.checkedId()
    seedWih=int(self.WihSeed.text())
    seedWho=int(self.WhoSeed.text())
    noOfHiddenLayer=int(self.noOfHiddenLayer.text())
    self.nN = nNPackage.neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate,activationFunction,randomMode,seedWih,seedWho,noOfHiddenLayer)
    self.nN.checkParameter()

    self.captureWidth=640
    self.captureHeight=120
    pass

    


