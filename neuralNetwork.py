import scipy.special
import numpy
#import toolFunctions

class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes,learningRate,activationFunction,randomMode,seedWih,seedWho,noOfHiddenLayer):
        self.iNodes = inputNodes
        self.oNodes = outputNodes
        self.noOfHiddenLayer=noOfHiddenLayer
        if(self.noOfHiddenLayer!=1 and self.noOfHiddenLayer!=2 and self.noOfHiddenLayer!=3):
            self.noOfHiddenLayer=1;       
        if(noOfHiddenLayer==1):
            self.hNodes = hiddenNodes
        elif(noOfHiddenLayer==2):
            self.hNodes = hiddenNodes
            self.hNodes2 = hiddenNodes
        elif(noOfHiddenLayer==3):
            self.hNodes = hiddenNodes
            self.hNodes2 = hiddenNodes
            self.hNodes3 = hiddenNodes
        else:
            self.hNodes = hiddenNodes
    
        self.lr = learningRate
        self.randomMode=randomMode
        self.seedWih=seedWih
        self.seedWho=seedWho
        self.totalEpochs=0
        if(randomMode==1):
            numpy.random.seed(seedWih)
            self.wih = numpy.random.rand(self.hNodes, self.iNodes)-0.5
            numpy.random.seed(seedWho)
            self.who = numpy.random.rand(self.oNodes, self.hNodes)-0.5
            if(noOfHiddenLayer==2):
                numpy.random.seed(6)
                self.whh = numpy.random.rand(self.hNodes, self.hNodes)-0.5
            elif(noOfHiddenLayer==3):
                numpy.random.seed(6)
                self.whh = numpy.random.rand(self.hNodes, self.hNodes)-0.5
                numpy.random.seed(7)
                self.whh2 = numpy.random.rand(self.hNodes, self.hNodes)-0.5               
        else:
            numpy.random.seed(seedWih)
            self.wih = numpy.random.normal (0.0, pow(self.hNodes,-0.5), (self.hNodes, self.iNodes))
            numpy.random.seed(seedWho)
            self.who = numpy.random.normal (0.0, pow(self.oNodes,-0.5), (self.oNodes, self.hNodes))
            if(noOfHiddenLayer==2):
                numpy.random.seed(6)
                self.whh = numpy.random.normal (0.0, pow(self.hNodes,-0.5), (self.hNodes, self.hNodes))
            elif(noOfHiddenLayer==3):
                numpy.random.seed(6)
                self.whh = numpy.random.normal (0.0, pow(self.hNodes,-0.5), (self.hNodes, self.hNodes))
                numpy.random.seed(7)
                self.whh2 = numpy.random.normal (0.0, pow(self.hNodes,-0.5), (self.hNodes, self.hNodes))
    
        switch = {
            1:lambda x:1/(1+numpy.exp(-x)),#sigmoid
            2:lambda x:2/(1+numpy.exp(-x)),#softsign
            3:lambda x:(numpy.fabs(x)+x)/2,#relu
            4:lambda x:(1-numpy.exp(-2*x))/(1+numpy.exp(-2*x)),#tanh
            5:lambda x:numpy.log(1+numpy.exp(x))#softplus
        }
        self.activation_function = switch[activationFunction]
        self.activationFunctionIndex=activationFunction
        #f'(x)
        #1  -numpy.exp(-x)/(1+numpy.exp(-x))^2
        #2  -2*numpy.exp(-x)/(1+numpy.exp(-x))^2
        #3  1/0
        #4  -4*numpy.exp(-2*x)/(1+numpy.exp(-2*x))^2
        #5
        #print(self.wih)
        pass
    def train(self,inputs_list, target_list,activateFunctionSelect):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        #print(inputs)
        #print(targets)
        if(self.noOfHiddenLayer==1):
            hidden_inputs = numpy.dot(self.wih,inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            final_inputs = numpy.dot(self.who,hidden_outputs)
            final_outputs = self.activation_function(final_inputs)
            output_errors = targets - final_outputs           
            hidden_errors = numpy.dot(self.who.T,output_errors)
            # all these Derivative results have been deleted the constant !in order to add... and you need to pay attention to them 
            if(activateFunctionSelect==1):#sigmoid ok
                fx1=numpy.exp(-final_inputs)/((1+numpy.exp(-final_inputs))*(1+numpy.exp(-final_inputs)))
                fx2=numpy.exp(-hidden_inputs)/((1+numpy.exp(-hidden_inputs))*(1+numpy.exp(-hidden_inputs)))            
            elif(activateFunctionSelect==2):#softsign  ok
                fx1=numpy.exp(-final_inputs)/((1+numpy.exp(-final_inputs))*(1+numpy.exp(-final_inputs)))
                fx2=numpy.exp(-hidden_inputs)/((1+numpy.exp(-hidden_inputs))*(1+numpy.exp(-hidden_inputs)))
            elif(activateFunctionSelect==3):#relu  ok
                final_inputs[final_inputs>0]=1
                hidden_inputs[hidden_inputs>0]=1
                final_inputs[final_inputs<0]=0
                hidden_inputs[hidden_inputs<0]=0
                fx1=final_inputs;
                fx2=hidden_inputs
            elif(activateFunctionSelect==4):#tanh  maybe we can do 
                fx1=numpy.exp(-2*final_inputs)/((1+numpy.exp(-2*final_inputs))*(1+numpy.exp(-2*final_inputs)))
                fx2=numpy.exp(-2*hidden_inputs)/((1+numpy.exp(-2*hidden_inputs))*(1+numpy.exp(-2*hidden_inputs)))            
            elif(activateFunctionSelect==5):#softplus ok
                fx1=numpy.exp(final_inputs)/(1+numpy.exp(final_inputs))
                fx2=numpy.exp(hidden_inputs)/(1+numpy.exp(hidden_inputs))
                pass
            self.who += self.lr * numpy.dot((output_errors * fx1),numpy.transpose(hidden_outputs))
            self.wih += self.lr * numpy.dot((hidden_errors * fx2),numpy.transpose(inputs))
        elif(self.noOfHiddenLayer==2):
            hidden_inputs = numpy.dot(self.wih,inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            hidden_inputs2 = numpy.dot(self.whh,hidden_outputs)
            hidden_outputs2 = self.activation_function(hidden_inputs2)
            final_inputs = numpy.dot(self.who,hidden_outputs2)
            final_outputs = self.activation_function(final_inputs)
            output_errors = targets - final_outputs
            hidden_errors2 = numpy.dot(self.who.T,output_errors)
            hidden_errors = numpy.dot(self.whh.T,hidden_errors2)
            # all these Derivative results have been deleted the constant !in order to add... and you need to pay attention to them 
            if(activateFunctionSelect==1):#sigmoid ok
                fx1=numpy.exp(-final_inputs)/((1+numpy.exp(-final_inputs))*(1+numpy.exp(-final_inputs)))
                fx2=numpy.exp(-hidden_inputs2)/((1+numpy.exp(-hidden_inputs2))*(1+numpy.exp(-hidden_inputs2)))
                fx3=numpy.exp(-hidden_inputs)/((1+numpy.exp(-hidden_inputs))*(1+numpy.exp(-hidden_inputs)))            
            elif(activateFunctionSelect==2):#softsign  ok
                fx1=numpy.exp(-final_inputs)/((1+numpy.exp(-final_inputs))*(1+numpy.exp(-final_inputs)))
                fx2=numpy.exp(-hidden_inputs2)/((1+numpy.exp(-hidden_inputs2))*(1+numpy.exp(-hidden_inputs2)))
                fx3=numpy.exp(-hidden_inputs)/((1+numpy.exp(-hidden_inputs))*(1+numpy.exp(-hidden_inputs)))
            elif(activateFunctionSelect==3):#relu  ok
                final_inputs[final_inputs>0]=1
                hidden_inputs[hidden_inputs>0]=1
                hidden_inputs2[hidden_inputs>0]=1
                final_inputs[final_inputs<0]=0
                hidden_inputs[hidden_inputs<0]=0
                hidden_inputs2[hidden_inputs<0]=0
                fx1=final_inputs;fx2=hidden_inputs2;fx3=hidden_inputs
            elif(activateFunctionSelect==4):#tanh  maybe we can do 
                fx1=numpy.exp(-2*final_inputs)/((1+numpy.exp(-2*final_inputs))*(1+numpy.exp(-2*final_inputs)))
                fx2=numpy.exp(-2*hidden_inputs2)/((1+numpy.exp(-2*hidden_inputs2))*(1+numpy.exp(-2*hidden_inputs2)))
                fx3=numpy.exp(-2*hidden_inputs)/((1+numpy.exp(-2*hidden_inputs))*(1+numpy.exp(-2*hidden_inputs)))                            
            elif(activateFunctionSelect==5):#softplus ok
                fx1=numpy.exp(final_inputs)/(1+numpy.exp(final_inputs))
                fx2=numpy.exp(hidden_inputs2)/(1+numpy.exp(hidden_inputs2))
                fx3=numpy.exp(hidden_inputs)/(1+numpy.exp(hidden_inputs))
                pass
            self.who += self.lr * numpy.dot((output_errors * fx1),numpy.transpose(hidden_outputs2))
            self.whh += self.lr * numpy.dot((hidden_errors2 * fx2),numpy.transpose(hidden_outputs))
            self.wih += self.lr * numpy.dot((hidden_errors * fx3),numpy.transpose(inputs))
            
        elif(self.noOfHiddenLayer==3):
            hidden_inputs = numpy.dot(self.wih,inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            hidden_inputs2 = numpy.dot(self.whh,hidden_outputs)
            hidden_outputs2 = self.activation_function(hidden_inputs2)
            hidden_inputs3 = numpy.dot(self.whh,hidden_outputs2)
            hidden_outputs3 = self.activation_function(hidden_inputs3)
            final_inputs = numpy.dot(self.who,hidden_outputs3)
            final_outputs = self.activation_function(final_inputs)
            output_errors = targets - final_outputs
            hidden_errors3 = numpy.dot(self.who.T,output_errors)
            hidden_errors2 = numpy.dot(self.whh2.T,hidden_errors3)
            hidden_errors = numpy.dot(self.whh.T,hidden_errors2)
            # all these Derivative results have been deleted the constant !in order to add... and you need to pay attention to them 
            if(activateFunctionSelect==1):#sigmoid ok
                fx1=numpy.exp(-final_inputs)/((1+numpy.exp(-final_inputs))*(1+numpy.exp(-final_inputs)))
                fx2=numpy.exp(-hidden_inputs2)/((1+numpy.exp(-hidden_inputs2))*(1+numpy.exp(-hidden_inputs2)))
                fx3=numpy.exp(-hidden_inputs3)/((1+numpy.exp(-hidden_inputs3))*(1+numpy.exp(-hidden_inputs3))) 
                fx4=numpy.exp(-hidden_inputs)/((1+numpy.exp(-hidden_inputs))*(1+numpy.exp(-hidden_inputs)))            
            elif(activateFunctionSelect==2):#softsign  ok
                fx1=numpy.exp(-final_inputs)/((1+numpy.exp(-final_inputs))*(1+numpy.exp(-final_inputs)))
                fx2=numpy.exp(-hidden_inputs2)/((1+numpy.exp(-hidden_inputs2))*(1+numpy.exp(-hidden_inputs2)))
                fx3=numpy.exp(-hidden_inputs3)/((1+numpy.exp(-hidden_inputs3))*(1+numpy.exp(-hidden_inputs3)))
                fx4=numpy.exp(-hidden_inputs)/((1+numpy.exp(-hidden_inputs))*(1+numpy.exp(-hidden_inputs)))
            elif(activateFunctionSelect==3):#relu  ok
                final_inputs[final_inputs>0]=1
                hidden_inputs[hidden_inputs>0]=1
                hidden_inputs2[hidden_inputs>0]=1
                hidden_inputs3[hidden_inputs>0]=1
                final_inputs[final_inputs<0]=0
                hidden_inputs[hidden_inputs<0]=0
                hidden_inputs2[hidden_inputs<0]=0
                hidden_inputs3[hidden_inputs<0]=0
                fx1=final_inputs;fx2=hidden_inputs2;fx3=hidden_inputs3;fx4=hidden_inputs
            elif(activateFunctionSelect==4):#tanh  maybe we can do 
                fx1=numpy.exp(-2*final_inputs)/((1+numpy.exp(-2*final_inputs))*(1+numpy.exp(-2*final_inputs)))
                fx2=numpy.exp(-2*hidden_inputs2)/((1+numpy.exp(-2*hidden_inputs2))*(1+numpy.exp(-2*hidden_inputs2)))
                fx3=numpy.exp(-2*hidden_inputs3)/((1+numpy.exp(-2*hidden_inputs3))*(1+numpy.exp(-2*hidden_inputs3)))    
                fx4=numpy.exp(-2*hidden_inputs)/((1+numpy.exp(-2*hidden_inputs))*(1+numpy.exp(-2*hidden_inputs)))                            
            elif(activateFunctionSelect==5):#softplus ok
                fx1=numpy.exp(final_inputs)/(1+numpy.exp(final_inputs))
                fx2=numpy.exp(hidden_inputs2)/(1+numpy.exp(hidden_inputs2))
                fx3=numpy.exp(hidden_inputs3)/(1+numpy.exp(hidden_inputs3))
                fx4=numpy.exp(hidden_inputs)/(1+numpy.exp(hidden_inputs))
                pass
            self.who += self.lr * numpy.dot((output_errors * fx1),numpy.transpose(hidden_outputs3))
            self.whh2 += self.lr * numpy.dot((hidden_errors3 * fx2),numpy.transpose(hidden_outputs2))
            self.whh += self.lr * numpy.dot((hidden_errors2 * fx3),numpy.transpose(hidden_outputs))
            self.wih += self.lr * numpy.dot((hidden_errors * fx4),numpy.transpose(inputs))
        pass
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        if(self.noOfHiddenLayer==1):
            hidden_inputs = numpy.dot(self.wih,inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            final_inputs = numpy.dot(self.who,hidden_outputs)
            final_outputs = self.activation_function(final_inputs)
        elif(self.noOfHiddenLayer==2):
            hidden_inputs = numpy.dot(self.wih,inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            hidden_inputs2 = numpy.dot(self.whh,hidden_outputs)
            hidden_outputs2 = self.activation_function(hidden_inputs2)
            final_inputs = numpy.dot(self.who,hidden_outputs2)
            final_outputs = self.activation_function(final_inputs)
        elif(self.noOfHiddenLayer==3):
            hidden_inputs = numpy.dot(self.wih,inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            hidden_inputs2 = numpy.dot(self.whh,hidden_outputs)
            hidden_outputs2 = self.activation_function(hidden_inputs2)
            hidden_inputs3 = numpy.dot(self.whh,hidden_outputs2)
            hidden_outputs3 = self.activation_function(hidden_inputs3)
            final_inputs = numpy.dot(self.who,hidden_outputs3)
            final_outputs = self.activation_function(final_inputs)    
            
        #hidden_outputs=numpy.insert(hidden_outputs,0,target)#need to modify query function by add target parameter
        #hidden_outputs=hidden_outputs.tolist()
        #toolFunctions.makeMNIST(hidden_outputs,'dstNeedToSAVE.csv')
        
        return final_outputs
        pass
    def checkParameter(self):
        print("self.iNodes",self.iNodes)
        print("self.oNodes",self.oNodes)
        print("hiddenNodes",self.hNodes)
        print("self.lr",self.lr)
        print("self random mode:",self.randomMode)
        print("self random seed Wih:",self.seedWih)
        print("self random seed Who:",self.seedWho)
        print("self activationFunctionIndex:",self.activationFunctionIndex)
        print("no. of layer:",self.noOfHiddenLayer)
        #print("self.wih",self.wih.shape,'  ',self.wih)
        #print("self.who",self.who.shape,'  ',self.who)
        pass

