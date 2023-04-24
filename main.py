import numpy as np 
import random
import matplotlib.pyplot as plt
import copy
import math
import tensorflow as tf
import tensorflow.keras as tk
#importing tensorflow for mnist dataset

class Model:

    def build(self,x,activation=None):
        if activation == None:
            self.activation=[]
            for i in range(0,len(x)):
                self.activation.append(0)
        else:
            self.activation = activation
        
        self.activation_array=['none','relu','softmax','sigmoid']

        b = []
        w = []
        for i in range(0,len(x)):
            
            if i > 0:

                w.append( np.random.rand( x[i-1],x[i]) ) 

                b.append( np.random.rand( x[i],1) ) 
                
        b = np.asarray(b,dtype=object)

        self.b = []

        for i in range(0,len(b)):
            self.b.append(b[i].T)
        self.w = w
        
        return self
    
    
    def activate(self,current,i):

        activation = self.activation_array[self.activation[i]]

        if(activation=='none'):
            return current
        
        elif(activation == 'relu') :

            r = np.where(current<0,0,current)
            
            return r
        
    
    def predict(self,input,pass_network= None,w=None,b = None):
        if w == None:
            w= self.w
        if b == None:
            b = self.b
        
        input = np.asarray(input)
        current = input

        z = []
        z.append(np.asarray([input]))

        activation_network = []
        input = self.activate(input,0)
        
        activation_network.append(np.asarray([input]))

        for i in range(0,len(self.b)):

            current = current.dot(self.w[i]) + self.b[i]
            
            z.append(current)

            current = self.activate(current,i)

            activation_network.append(current)

        if pass_network==1:

            return (current,z,activation_network)
        
        return current 

    
    def train(self, x, y , lr = 0.0000001,epochs = 1,batch_size = 10, op = 'pred' ):

        for epoch in range(0,epochs):
            el =0

            for batch in range (0,int(len(x)/batch_size)):
                
                bl = 0
                inp = []
                
                out = []

                for i in range(batch*batch_size,batch_size+batch*batch_size):
                    inp.append(x[i])
                    out.append(y[i])

                inp = np.asarray(inp)
                out = np.asarray(out)                
                bl = self.back_prop(inp,out,lr,op)
                print('Epochs : ',epoch+1,',Batch_loss : ',bl,end = '\r')
                el +=bl
                    
            inp = []
            out = []
            l = 0
            for i in range(0,len(x)%batch_size):

                ind = i+int(len(x)/batch_size)*batch_size

                inp.append(x[ind])
                out.append(y[ind])
          
            inp = np.asarray(inp)
            out = np.asarray(out)
            bl = self.back_prop(inp,out,lr,op)
            print('Epochs : ',epoch+1,'Batch_loss : ',bl,end = '\r')
            el +=bl
            print('Epoch : ',epoch+1,'Loss : ',el/len(x))
    
    def delta_activation(self,dzi,i):

        activation = self.activation_array[self.activation[i]]

        if(activation =='none'):

            val = np.where( (dzi >0) | (dzi<=0),1,dzi)
            return val
        
        elif(activation == 'relu'):

            val = np.where((dzi > 0),1,dzi)

            return val


    def back_prop(self,x,y,lr,op):
        if len(x) == 0:
            return 0
        
        else :
            dw = {}
            db = {}
            l = 0 
            lr_rate = lr
                    
            for i in range(0 , len(x)):

                pred,z,a = self.predict(x[i],pass_network=1)
                dif = pred - y[i]
                dzi = np.asarray(dif)
                loss = np.sum(np.abs(dzi[0]))
                
                if op =='gen':
                    lr_rate = lr/pow(loss,self.sigmoid(loss))

                l+=loss

                for layer in range(len(self.b)-1,-1,-1):

                    dw[layer] = ((np.asarray(a[layer]).T).dot(dzi) /len(x))
                    db[layer] = (dzi /len(x))
                    dai = dzi.dot(np.asarray(self.w[layer]).T)
                    dzi = dai  * self.delta_activation(z[layer],layer) 
                    self.b[layer] = (np.asarray([self.b[layer]]) - (lr_rate)*db[layer])[0]
                    self.w[layer] = self.w[layer]  - (lr_rate)*(dw[layer])

            return l/len(y)


    def mean_squared_error(self,x,y):
        pre = self.predict(x)
        loss = pre-y
        ls = 0
        for i in range(0,len(loss)):
            ls  += abs(loss[i])
        
        return ls
    
    def get_bias(self):
        return self.b
    

    def get_weights(self):
        return self.w
    
    
    def sigmoid(self,x):
        e = math.e
        sig_x = 1/(1+pow(e,-x))
        return (sig_x)

    
(img , label ), (o , p )= tf.keras.datasets.mnist.load_data()
img = img.reshape(-1,28*28)
img = img /255
s=[]
for i in range(0,len(label)):
    d = np.zeros(10)
    d[label[i]]=1000
    s.append(d)
s= np.asarray(s)

lb =[]
for i in range (0,len(label)):
    lb.append(np.asarray([label[i]]))

lb = np.asarray(lb)
print(lb.shape)

cls = Model().build([28*28,11,10,10],activation=[1,0,0,0])

# cls.train( img[0:10000] , s[0:10000] , lr=0.0001 , batch_size=100, op = 'pred',epochs = 10)

# I didn't build the model to print the accuracy, so i made the below snippet to display the accuracy 

e = 0
for epoch in range(1,10):
    e +=1
    # training the model on 10,000 images , why ?
    # obviously to save time ;)
    cls.train( img[0:10000] , s[0:10000] , lr=0.0001 , batch_size=100, op = 'pred',epochs = 1)
    c= 0
    for i in range(1,100):
        if( np.argmax(s[i]) == np.argmax(cls.predict(img[i])) ):
            c+=1
    print('Accuracy : ',c*100.0/i)


c= 0
res = cls.predict(img)
for i in range(0,len(img)):
    if( np.argmax(s[i]) == np.argmax(res[i]) ):
        c+=1
print('\nResults on 60,000 image dataset\n','Total images : ',i,'\nCorrect predictions : ',c,'\nAccuracy : ',c*100.0/i)

#change the value image_index to predict the model
image_index = 180
test_image = img[image_index:image_index+1]
result = cls.predict([test_image])
temp_image = test_image.reshape(28,28)
print('\nprediction : ',np.argmax(result))
plt.imshow(temp_image)
plt.show()

