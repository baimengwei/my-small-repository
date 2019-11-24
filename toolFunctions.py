import matplotlib.pyplot as plt
from skimage import io,color,transform,filters,morphology,exposure
import skimage.morphology as sm
import numpy as np
from skimage import measure
import numpy
import csv

#many pictures display
#inputs:：image：1+28X28=785 dot each line
#      prediction：if it's null then not display
#      option：enable if the number array don't include ','
#      num：number of the picture
#      labelEnable: for show label and choose the way for split the number
#outputs:    show picture
#others :the max of the num is 50
def plotImageLabelsPrediction(images,prediction=[],labelEnable=1,option=0,num=10):
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(12,24)
    if num>50 : num = 50
    for i in range(0,num):
        title = ""
        ax = plt.subplot(10,5,1+i)
        if(option==0):
            all_values = images[i].split(',')
        else:
            all_values = images[i]
        if(labelEnable == 1):
            image_array = numpy.asfarray( all_values [1:]).reshape((28,28))
            title = "lable="+str(all_values[0])
        else:
            image_array = numpy.asfarray(all_values).reshape((28,28))
              
        ax.imshow(image_array,cmap='binary')
        if len(prediction)>0:
            title+=" predict="+str(prediction[i])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
    
    plt.show()
    plt.subplots_adjust(hspace = 1)
    
    pass

#make picture light
#inputs:pic: a gray picture
#outputs: a light picture
#others:    None
def pictureBright(pic):
    img_gray = pic
    img_gray=exposure.adjust_gamma(img_gray,0.2)#提升亮度
    #plt.figure()
    #plt.imshow(img_gray, cmap = 'Greys_r')   
    img_gray = exposure.equalize_hist(img_gray,nbins=5)#直方化
    return img_gray
    pass

#find the edge of the picture
#inputs:    a gray picture：multiple is 0 for a single digital,threshold is for recognize the block of digital
#outputs:   coordinate and original data
#others     None
def pictureFindEdge(pic,multiple=0,threshold=0.5):
    img_gray = pic
    img_gray_bin=(img_gray<threshold)*1#二值化
    chull = morphology.convex_hull_object(img_gray_bin)#框定连通区域
    labels=measure.label(chull,connectivity=2)  # 8连通区域标记
    point_l=np.zeros(labels.max()+1,dtype=np.int16);point_r=np.zeros(labels.max()+1,dtype=np.int16)
    point_t=np.zeros(labels.max()+1,dtype=np.int16);point_b=np.zeros(labels.max()+1,dtype=np.int16)
    for i in range(labels.max()+1):
        point_l[i] = labels.shape[1];point_t[i] = labels.shape[0]
    for i in range(labels.shape[0]):#宽
        for j in range(labels.shape[1]):#长
            if labels[i][j] != 0:
                if point_l[labels[i][j]] > j :
                    point_l[labels[i][j]] = j
                if point_r[labels[i][j]] < j :
                    point_r[labels[i][j]] = j
                if point_t[labels[i][j]] > i :
                    point_t[labels[i][j]] = i
                if point_b[labels[i][j]] < i :
                    point_b[labels[i][j]] = i
    point_l=point_l.tolist();point_l.pop(0);point_r=point_r.tolist();point_r.pop(0)
    point_t=point_t.tolist();point_t.pop(0);point_b=point_b.tolist();point_b.pop(0)
    if(multiple==0):
        point_l=[min(point_l)];point_r=[max(point_r)];point_t=[min(point_t)];point_b=[max(point_b)];
    return point_l,point_r,point_t,point_b,labels
#l,r,t,b,version=picture_find_edge(io.imread('./myhandwritting5/2_5.png'))
#io.imshow(version)
#print(l,r,t,b)


#add empty in the picture
#inputs:    a gray picture , ratio of height and width
#return:    a gray picture
#others:None
#
def addEmpty(img_gray_cut,height_param=4,width_param=2):
    size_width = img_gray_cut.shape[1]#给图像添加空白
    size_height = img_gray_cut.shape[0]
    #print(img_gray_cut.shape)
    img_gray_cut=img_gray_cut.tolist()
    for i in range(int(size_height/height_param)):#增加x行x列空白。增高：x:2+x
        img_gray_cut.insert(0,np.zeros(size_width))
        img_gray_cut.append(np.zeros(size_width))
    img_gray_cut=np.array(img_gray_cut)
    
    size_width = img_gray_cut.shape[1]
    size_height = img_gray_cut.shape[0]
    addWidthParam=size_width/width_param
    if(size_width<size_height/5):#if the digital is too thin
        addWidthParam=size_height/2
        pass
    for i in range(int(addWidthParam)):#增加x行x列空白。增宽：x:2+x
        img_gray_cut=np.insert(img_gray_cut,0,0,axis=1)
        img_gray_cut=np.append(img_gray_cut,np.array([list(np.zeros(size_height))]).T,axis=1)
    return img_gray_cut
    pass


#handle picture include find edge,cut,make them clear and so on
#input：a gray picture and option used to handle picture after cut.
#output:：28X28 dot within 0-255 each one
#others: None
def handlePicture(pic_str,option=3):
    
    img_gray = pictureBright(pic_str)
    point_l,point_r,point_t,point_b,lables=pictureFindEdge(img_gray)#for single digital
    
    img_gray_cut = img_gray[point_t[0]:point_b[0],point_l[0]:point_r[0]]#cut the picture
    img_gray_cut = exposure.rescale_intensity(img_gray_cut,in_range=(0,1),out_range=(0,255))#数值从0-1扩展到0-255
    img_gray_cut=255-img_gray_cut#图像颜色变更
    
    #img_gray_cut=transform.resize(img_gray_cut,(40,20),mode='constant')#转为80X80点图用于居中图像
    
    img_gray_cut=addEmpty(img_gray_cut,4,2)#增加空白。高2:3  宽1:2

    img_dst=transform.resize(img_gray_cut,(28,28),mode='constant')#转为28X28点图获得目标图
   
    if(option == 0):#不变
        img_dst =img_dst
    elif(option ==1):#加粗滤波
        img_dst=sm.erosion(img_dst,sm.square(2))
    elif(option ==2):#加粗滤波
        img_dst=sm.erosion(img_dst,sm.square(2));img_dst=filters.gaussian(img_dst,sigma=1)
    elif(option ==3):#加粗滤波
        img_dst=sm.erosion(img_dst,sm.square(1));img_dst=filters.gaussian(img_dst,sigma=1)
    elif(option ==4):#加粗减粗滤波
        img_dst=sm.erosion(img_dst,sm.square(2));img_dst=sm.dilation(img_dst,sm.square(1));img_dst=filters.gaussian(img_dst,sigma=1)

    img_dst=img_dst.astype(np.float32)
    #plt.imshow(img_dst, cmap='binary')
    return img_dst
#img_dst = handle_picture('hope9.png')
#a=handle_picture('youtemp.png')

#collect data from my own dataPath for train
#inputs: None but need filePath
#outputs:test__List of all data
#others:1+784 each line
def makeDatabase():#自定义图像建立数据集，样式和MNIST相同
    test__list=[]
    for i in range(73):
        for j in range(10):
            img_rgb = io.imread('./myhandwritting_train/'+str(i+1)+'_'+str(j)+'.png')
            img_gray = color.rgb2gray(img_rgb)
            img_data = handlePicture(img_gray)
            img_data = img_data.reshape(784)
            img_data = img_data.tolist()
            img_data.insert(0,j)
            test__list.append(img_data)
    return test__list

#sort the point 
#inputs:    point_left,point_right,point_top,point_buttom
#outputs:   sorted arrays
#others:    None
def sortPoint(point_l,point_r,point_t,point_b):
    for i in range(0,len(point_l)-1):
        point_l_min_index=i
        for j in range(i+1,len(point_l)):
            if(point_l[point_l_min_index]>point_l[j]):
                point_l_min_index=j
        point_l[point_l_min_index],point_l[i]=point_l[i],point_l[point_l_min_index]
        point_r[point_l_min_index],point_r[i]=point_r[i],point_r[point_l_min_index]
        point_t[point_l_min_index],point_t[i]=point_t[i],point_t[point_l_min_index]
        point_b[point_l_min_index],point_b[i]=point_b[i],point_b[point_l_min_index]
        #swap
    pass


#delete point which is small or on the edge
#inputs:    point_left,point_right,point_top,point_buttom and threshold for small point
#outputs:   altered arrays
#others:    None
def deleteSmallPoint(point_l,point_r,point_t,point_b,threshold=15):
    popIdx=0
    i=0
    length=len(point_l)
    while(1):
        if(popIdx+i==length):
            break;    
        if(point_l[i]==0 or point_r[i]==0 or point_t[i]==0 or point_b[i]==0 or point_r[i]-point_l[i]+point_b[i]-point_t[i]<threshold):
            point_l.pop(i);point_r.pop(i);point_t.pop(i);point_b.pop(i);
            popIdx+=1;
            i-=1;
        i+=1
    i=i+popIdx-len(point_l)#恢复初始值，直接赋值失败！
    pass

#store csv file 
#inputs: one line data and filePath plus name
#outputs:None
#others: None
def storFile(data,fileName,method='a'):
    with open(fileName,method,newline ='') as f:
        mywrite = csv.writer(f)
        mywrite.writerow(data)
        pass
    pass

#make my mnist database which is changed on my way for butter recognize
#input: mnist data of each line
#output:training data in [1:] for further training
#others:a little longer
def makeAlterMNIST(all_values):
    pic = numpy.asfarray( all_values [1:])
    pic = transform.resize(pic,(28,28),mode='constant')
    point_l,point_r,point_t,point_b,lables=pictureFindEdge(pic,multiple=0,threshold=0.5)
    img_gray_cut = pic[point_t[0]:point_b[0],point_l[0]:point_r[0]]
    img_gray_cut = addEmpty(img_gray_cut,height_param=4,width_param=2)
    img_gray_cut = transform.resize(pic,(28,28),mode='constant')
    img_dst = img_gray_cut.reshape(784)
    img_dst = img_dst.astype(int)#change to int
    img_dst = img_dst.tolist()
    img_dst.insert(0,all_values[0])
    #print(img_dst)
    storFile(img_dst,'dstNeedToSAVE.csv')
    img_dst = np.asfarray(img_dst[1:])
    return img_dst
    pass

#make my database formate csv
#input: mnist data of each line
#output:None
#others:generate a csv file
def makeMNIST(all_values,fileName):
    #print(img_dst)
    storFile(all_values,fileName)
    pass

def storeWritting(all_data,fileName):
    i=0
    for data in all_data:
        data=data.astype(int)
        data=data.tolist()
        data.insert(0,i)
        i=i+1
        storFile(data, fileName)
        pass
    pass

#draw a 3D map with axis values
#def draw3D(X,Y,Z):
#    figure = plot.figure()
#    axes = Axes3D(figure)
#    #X = np.arange(-10, 10, 0.25)
#    #Y = np.arange(-10, 10, 0.25)
#    X, Y = np.meshgrid(X, Y)
#    #Z = X+Y
#    axes.plot_surface(X, Y, Z,cmap='rainbow')
#    plot.show()    
#    pass








