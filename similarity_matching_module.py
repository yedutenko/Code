###Third-party implementation of the similarity-matching approach for estimation of motion 
###Original paper - https://proceedings.neurips.cc/paper/2019/hash/dab1263d1e6a88c9ba5e7e294def5e8b-Abstract.html

###Import Basic packages which you will need
import numpy as np
import scipy as sci
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import math
import random
import pandas as pd


###Define Relu and whitening functions
###ReLu:
def relu(x):
    ###Rectification
    y=np.zeros((len(x),1))
    for i in range(len(x)):
         y[i]=max(0,x[i])
    return (y)
###ZCA whitening:
def zca_whiten(X,EPS=1e-6):
  ###Zero-phase component analysis whitening
    assert(X.ndim == 2)
    ###EPS - is a coefficient, which prevents high-order principal components from explosion.
    ###Note, that EPS lower than 1e-2 leads to explosion in learning weighs.

    #   covariance matrix
    X=X-np.mean(X)
    X=X/np.std(X)
    cov = np.dot(X.T, X)/(len(X)-1)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1. / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)

    X_white = np.dot(X, W)
    X_white=X_white-np.mean(X_white)
    X_white=X_white/np.std(X_white)

    return X_white

###Functions to generate stimuli.
###1D stimulus where at each step we randomly generate texture and than shift by subpixel translation
def Translation_Generator(NumEx,mu, sigma,ImageSize,UpSample,Filter,TrLim,rf_size,set_whiten=1):
    ###The idea is:
    #1.to create an image, 
    #2.upsamle it to achieve subpizel resolution,
    #3. translate it,
    #4.downsample back a
    #5.  cut out small image with size rf_size to avoid edge-effects
    ###Parameters
    #NumEx - number of desired examples
    #mu -   mean of the image
    #sigma - standard deviation
    #Filter - filter parameters for low-passing
    #TrLim - Maximal magnitude of ranslation in subpixel*UpSample
    #set_whiten - whether one wants to whiten image set (recommended)

    # The rest I held to be self-evident
    Original=[]
    Transformed=[]
    TransformedHS=[]
    OriginalHS=[]
    Magnitude=[]
    for i in range (NumEx):
        ##Create Image
        Image=np.random.normal(mu,sigma,ImageSize[0]*ImageSize[1])
        Image=np.reshape(Image,[int(Image.shape[0]),1])
        ImageF=sci.fft.fft(Image)
        #Filter Image
        ImageF=ImageF*Filter
        ImageF=np.reshape(ImageF,[ImageSize[0]])
        ImageOr=np.real(sci.fft.ifft(ImageF))
        ImageOr=np.reshape(ImageOr,ImageSize)
        ImageOr=ImageOr*np.std(Image)/np.std(ImageOr)
        ImageOr=ImageOr-np.mean(ImageOr)
        #Translate Image
        dsize2=(1,ImageSize[0]*UpSample)
        ImageO=cv2.resize(ImageOr,dsize=dsize2,interpolation=cv2.INTER_LINEAR)
        V=np.random.randint(low=-TrLim,high=TrLim+1,size=1)*0    ###vertical shift
        H=np.random.randint(low=-TrLim,high=TrLim+1,size=1)   ###horizontal shift
        TrM=np.float32([[1,0,V],[0,1,H]])
        ImageT=cv2.warpAffine(ImageO,TrM,dsize=dsize2)
        #Downsample back to standard size
        ImageTL=cv2.resize(ImageT,dsize=(1,ImageSize[0]))
        ImageOr=cv2.resize(ImageO,dsize=(1,ImageSize[0]))
        ImageOr=ImageOr[int(ImageSize[0]/2):int(rf_size+ImageSize[0]/2),
        0]
        ImageOr=ImageOr-np.mean(ImageOr)
        ImageOr=ImageOr/np.std(ImageOr)
        ImageTL=ImageTL[int(ImageSize[0]/2):int(rf_size+ImageSize[0]/2),
        0]
        ImageTL=ImageTL-np.mean(ImageTL)
        ImageTL=ImageTL/np.std(ImageTL)
        #ImageOr=zca_whiten(ImageOr)
        #ImageTL=zca_whiten(ImageTL)
        Original.append(ImageOr)
        Transformed.append(ImageTL)
        Magnitude.append([H,V])
        OriginalHS.append(ImageO)
        TransformedHS.append(ImageT)
    Original=np.array(Original)
    Transformed=np.array(Transformed)
    if set_whiten==1:
        Original=zca_whiten(Original)
        Transformed=zca_whiten(Transformed)
    Original=np.reshape(Original,[NumEx,rf_size,1])
    Transformed=np.reshape(Transformed,[NumEx,rf_size,1])
    ImagePair=np.concatenate([Original,Transformed],axis=2)

    #ImagePair=[Original,Transformed]
    return ImagePair,Magnitude

###1D stimulus when we first create correlated world and than sample from it
def Translation_World(signal_length,mu, sigma,smooth_const,smooth_std,MaxT,Jump_Translate, rf_size,noise_std=0):
    ###The idea is:
    #1.Create correlated 1D world in superpixel (i.e.*UpSample) resolutionn
    #2. Pick small, downsampled images with size rf_size at position t and t+translation
    #3. Whiten set

    #mu - mean of the signal
    #sigma - std of the signal
    #MaxT- maximal translation
    #Jump_Translate - down/up-sampling factor
    #rf_size - receptive field of EMD i.e. number of pixels
    #The name of the rest of the parameters are self-explanatory, I believe
    
    ###Creation of the signal
    u=np.random.normal(mu,sigma,[signal_length,1])
    window = signal.windows.gaussian(smooth_const, std=smooth_std)
    u_smooth=np.convolve(window,u[:,0],mode='same')
    u_smooth=u_smooth.reshape(signal_length,1)
    plt.figure()
    plt.plot(u[1:1000],'b')
    plt.plot(u_smooth[1:1000],'r')
    NumEx=round((signal_length-(rf_size-1)-MaxT)/(Jump_Translate*smooth_const))
    
  # Concatenation of Variables.
    x_t_concat=[]
    x_t_1_concat=[]

 #Range of Translation Steps
    Translation_steps=np.arange(-MaxT,MaxT+1,1)
    Translation_steps=list(Translation_steps)
    Translation_steps.remove(0)

 # Image and derivative
    Magnitude=[]
    for t in range(MaxT,NumEx):
        Translation=random.choice(Translation_steps)
        x_t_1=u_smooth[t:t+Jump_Translate*rf_size-1:Jump_Translate]
        x_t=u_smooth[t+Translation:t+Jump_Translate*rf_size-1+Translation:Jump_Translate]
        x_t_concat.append(x_t)
        x_t_1_concat.append(x_t_1)
        Magnitude.append(Translation)

    NumEx=NumEx-MaxT

    x_t_concat=np.array(x_t_concat)
    noise=np.random.normal(0,noise_std,x_t_concat.shape)
    x_t_concat=x_t_concat+noise
    x_t_concat=zca_whiten(x_t_concat[:,:,0])
    x_t_concat=np.reshape(x_t_concat,[NumEx,rf_size,1])

    x_t_1_concat=np.array(x_t_1_concat)
    noise=np.random.normal(0,noise_std,x_t_concat.shape)
    x_t_1_concat=x_t_1_concat+noise
    x_t_1_concat=zca_whiten(x_t_1_concat[:,:,0])
    x_t_1_concat=np.reshape(x_t_1_concat,[NumEx,rf_size,1])   
    ImagePair=np.concatenate([x_t_1_concat,x_t_concat],axis=2)     
    
    return ImagePair,Magnitude, NumEx  

###2D translations where at each step we generate correlated texture and than translate it 
#  
#A.2D translation where at each step we move either vertically or horizontally
def Translation_Generator_2D_Train (NumEx,mu,sigma,ImageSize,rf_size, smooth_std,UpSample, TrLim, smoothing=1,image_whiten=0,set_whiten=1):
    #The idea is:
    #1. Create correlated texture
    #.2. UpSample it
    #3. Translate it
    #4. Downsample back 
    #5. Whiten

    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    ##ImageSize - size of the translated image, from each we latter take chunk with size rf_size
    #TrLim - Maximal Amplitude of translation
    #smoothing - decides whether or not make texture correlated
    #image_whiten - determines whether separate textures should be whiten
    #set_whiten - determines whether the entire stimuli set should be whiten 
    from scipy.ndimage import gaussian_filter
    Original=[]
    Translated=[]
    Magnitude=[]
    
    for i in range (NumEx):
        Image=np.random.normal(mu,sigma,ImageSize)
###smoothing
        if smoothing==1:
            Image=gaussian_filter(Image,smooth_std)
        Image=Image-np.mean(Image)
        Image=Image/np.std(Image)
        Image=cv2.resize(Image,dsize=ImageSize*UpSample,interpolation=cv2.INTER_LINEAR)
        
        it=np.random.randint(low=0,high=2)
        if it==0:
            V=np.random.randint(low=-TrLim,high=TrLim+1)   ###vertical shift
            H=0
        elif it==1:
            V=0
            H=np.random.randint(low=-TrLim,high=TrLim+1)
        TrM=np.float32([[1,0,V],[0,1,H]])

        ImageT=cv2.warpAffine(Image,TrM,dsize=ImageSize*UpSample)
        
        ImageO=cv2.resize(Image,dsize=ImageSize,interpolation=cv2.INTER_LINEAR)
        ImageT=cv2.resize(ImageT,dsize=ImageSize,interpolation=cv2.INTER_LINEAR)
        
        center=int(np.round(ImageSize[0]/2))

        ImageO=ImageO[center:center+rf_size,center:center+rf_size]         
        ImageT=ImageT[center:center+rf_size,center:center+rf_size]  

        ImageO=ImageO-np.mean(ImageO)
        ImageO=ImageO/np.std(ImageO)
        ImageT=ImageT-np.mean(ImageT)
        ImageT=ImageT/np.std(ImageT)

        if image_whiten==1:
            ImageO=zca_whiten(ImageO)
            ImageT=zca_whiten(ImageT)  

        Original.append(ImageO)
        Translated.append(ImageT)
        Magnitude.append([V,H])

    Original=np.array(Original)
    Translated=np.array(Translated)
    if set_whiten==1:
        Original=np.reshape(Original,[NumEx,rf_size**2])
        Original=zca_whiten(Original)
        Translated=np.reshape(Translated,[NumEx,rf_size**2])
        Translated=zca_whiten(Translated)
    Original=np.reshape(Original,[NumEx,rf_size**2,1])
    Translated=np.reshape(Translated,[NumEx,rf_size**2,1])
    ImagePair=np.concatenate([Original,Translated],axis=2)
    
    return ImagePair, Magnitude

#B. 2D translaton where eat each step we move both vertically and horizontally
def Translation_Generator_2D_Valid (NumEx,mu,sigma,ImageSize,rf_size, smooth_std,UpSample, TrLim, smoothing=1,image_whiten=0,set_whiten=1):
    #The idea is:
    #1. Create correlated texture
    #.2. UpSample it
    #3. Translate it
    #4. Downsample back 
    #5. Whiten

    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    ##ImageSize - size of the translated image, from each we latter take chunk with size rf_size
    #TrLim - Maximal Amplitude of translation
    #smoothing - decides whether or not make texture correlated
    #image_whiten - determines whether separate textures should be whiten
    #set_whiten - determines whether the entire stimuli set should be whiten 
    from scipy.ndimage import gaussian_filter
    Original=[]
    Translated=[]
    Magnitude=[]
    
    for i in range (NumEx):
        Image=np.random.normal(mu,sigma,ImageSize)
###smoothing
        if smoothing==1:
            Image=gaussian_filter(Image,smooth_std)
        Image=Image-np.mean(Image)
        Image=Image/np.std(Image)
        Image=cv2.resize(Image,dsize=ImageSize*UpSample,interpolation=cv2.INTER_LINEAR)
        
        V=np.random.randint(low=-TrLim,high=TrLim+1)   ###vertical shift
        H=np.random.randint(low=-TrLim,high=TrLim+1)
        TrM=np.float32([[1,0,V],[0,1,H]])
        
        ImageT=cv2.warpAffine(Image,TrM,dsize=ImageSize*UpSample)
        
        ImageO=cv2.resize(Image,dsize=ImageSize,interpolation=cv2.INTER_LINEAR)
        ImageT=cv2.resize(ImageT,dsize=ImageSize,interpolation=cv2.INTER_LINEAR)
        
        center=int(np.round(ImageSize[0]/2))

        ImageO=ImageO[center:center+rf_size,center:center+rf_size]         
        ImageT=ImageT[center:center+rf_size,center:center+rf_size]  

        ImageO=ImageO-np.mean(ImageO)
        ImageO=ImageO/np.std(ImageO)
        ImageT=ImageT-np.mean(ImageT)
        ImageT=ImageT/np.std(ImageT)

        if image_whiten==1:
            ImageO=zca_whiten(ImageO)
            ImageT=zca_whiten(ImageT)  

        Original.append(ImageO)
        Translated.append(ImageT)
        Magnitude.append([V,H])

    Original=np.array(Original)
    Translated=np.array(Translated)
    if set_whiten==1:
        Original=np.reshape(Original,[NumEx,rf_size**2])
        Original=zca_whiten(Original)
        Translated=np.reshape(Translated,[NumEx,rf_size**2])
        Translated=zca_whiten(Translated)
    Original=np.reshape(Original,[NumEx,rf_size**2,1])
    Translated=np.reshape(Translated,[NumEx,rf_size**2,1])
    ImagePair=np.concatenate([Original,Translated],axis=2)
    
    return ImagePair, Magnitude

###2D translations where we create correlated world and than sample from it

def Translation_Generator_2D_World (NumEx,mu,sigma,ImageSize,rf_size, smooth_std,UpSample, TrLim, smoothing=1,image_whiten=0,set_whiten=1):
  #The idea is:   
    #1.Create correlated 2D world in superpixel (i.e.*UpSample) resolutionn
    #2. Pick small, downsampled images with size rf_size at position t and t+translation
    #3. Whiten
    ###At the moment is only for movements either horixontally or vertically, but it can be easily changed
    ###Parameters
    #The idea is:
    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    ##ImageSize - size of the translated image, from each we latter take chunk with size rf_size
    #TrLim - Maximal Amplitude of translation
    #smoothing - decides whether or not make texture correlated
    #image_whiten - determines whether separate textures should be whiten
    #set_whiten - determines whether the entire stimuli set should be whiten 
    from scipy.ndimage import gaussian_filter
    Original=[]
    Translated=[]
    Magnitude=[]
    
    Image=np.random.normal(mu,sigma,ImageSize)
    ###smoothing
    if smoothing==1:
        Image=gaussian_filter(Image,smooth_std)
        Image=Image-np.mean(Image)
        Image=Image/np.std(Image)


    for i in range (NumEx):

        Corner=np.random.randint(TrLim+1,Image.shape[0]-rf_size*UpSample-TrLim-2,2)

        it=np.random.randint(low=0,high=2)
        if it==0:
            V=np.random.randint(low=-TrLim,high=TrLim+1)   ###vertical shift
            H=0
        elif it==1:
            V=0
            H=np.random.randint(low=-TrLim,high=TrLim+1)



        ImageO=Image[Corner[0]:Corner[0]+rf_size*UpSample:UpSample,Corner[1]:Corner[1]+rf_size*UpSample:UpSample]
        ImageT=Image[Corner[0]+V:Corner[0]+V+rf_size*UpSample:UpSample,Corner[1]+H:Corner[1]+H+rf_size*UpSample:UpSample]

        if image_whiten==1:
            ImageO=zca_whiten(ImageO)
            ImageT=zca_whiten(ImageT)  

        Original.append(ImageO)
        Translated.append(ImageT)
        Magnitude.append([V,H])

    Original=np.array(Original)
    Translated=np.array(Translated)
    if set_whiten==1:
        Original=np.reshape(Original,[NumEx,rf_size**2])
        Original=zca_whiten(Original)
        Translated=np.reshape(Translated,[NumEx,rf_size**2])
        Translated=zca_whiten(Translated)
    Original=np.reshape(Original,[NumEx,rf_size**2,1])
    Translated=np.reshape(Translated,[NumEx,rf_size**2,1])
    ImagePair=np.concatenate([Original,Translated],axis=2)
    
    return ImagePair, Magnitude

###Stream of translated images in correlated 2D world

def Translation_Generator_2D_Stream(NumEx,mu,sigma,ImageSize,smooth_std,rf_size,UpSample,TrLim,smoothing=1,image_whiten=0,set_whiten=1):
     ##Here we augmented original paper by Bahroun et al. by adding "stream of images" as a stimuli.
     ##In the original paper stimuli at each training step consisted of pair image-translated image.
     ###Here we generate sequence of images and train our network to take an image at time t, compare it 
     ### with image at moment t-1 and find transformation operator.
     ###At the moment it does only vertical or horizontal at each step, but it can be easily changed to simultaneous translations
     ###So the idea is:
     #1. Create correlated, superpixel 2D world
     #2. Update coordinates
     #3. Take downsampled (size=rf_size) image and coordinates
     #4. Whiten
     ###Parameters
    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    ##ImageSize - size of the translated image, from each we latter take chunk with size rf_size
    #TrLim - Maximal Amplitude of translation
    #smoothing - decides whether or not make texture correlated
    #image_whiten - determines whether separate textures should be whiten
    #set_whiten - determines whether the entire stimuli set should be whiten 
    from scipy.ndimage import gaussian_filter
    Image_Stream=[]
    Magnitude=[]
    
    Image=np.random.normal(mu,sigma,ImageSize)
    ###smoothing
    if smoothing==1:

        Image=gaussian_filter(Image,smooth_std)
        Image=Image-np.mean(Image)
        Image=Image/np.std(Image)
  ###Starting coordinates
    center=int(np.round((ImageSize[0]-rf_size*UpSample)/2))
    Coord=np.array((center,center))
    Image_Stream.append(Image[center:center+rf_size*UpSample:UpSample,
            center:center+rf_size*UpSample:UpSample])
    ###Determine translation steps
    for i in range (NumEx):

        it=np.random.randint(low=0,high=2)
        if it==0:
            V=np.random.randint(low=-TrLim,high=TrLim+1)   ###vertical shift
            H=0
        elif it==1:
            V=0
            H=np.random.randint(low=-TrLim,high=TrLim+1)

        Coord=Coord+(H,V)
       ##Samle image
        Image_T=Image[Coord[0]:Coord[0]+UpSample*rf_size:UpSample,
        Coord[1]:Coord[1]+UpSample*rf_size:UpSample]
        
        if image_whiten==1:

            Image_T=zca_whiten(Image_T)


        Image_Stream.append(Image_T)
        Magnitude.append([V,H])
    
    Image_Stream=np.array(Image_Stream)
    Magnitude=np.array(Magnitude)

    Image_Stream=np.reshape(Image_Stream,[NumEx+1,rf_size**2])
    
    if set_whiten==1:
        Image_Stream=zca_whiten(Image_Stream)

    Image_Stream=np.reshape(Image_Stream,[NumEx+1,rf_size**2,1])  

    return Image_Stream,Magnitude

def Translation_Generator_2D_Stream_2(NumEx,L,mu,sigma,ImageSize,smooth_std,rf_size,UpSample,TrLim,smoothing=0,image_whiten=0,set_whiten=1):
     ##Here we augmented original paper by Bahroun et al. by adding "stream of images" as a stimuli.
     ##In the original paper stimuli at each training step consisted of pair image-translated image.
     ###Here we generate sequence of images and train our network to take an image at time t, compare it 
     ### with image at moment t-1 and find transformation operator.
     ###At the moment it does only vertical or horizontal at each step, but it can be easily changed to simultaneous translations
     ###So the idea is:
     #1. Create correlated, superpixel 2D world
     #2. Update coordinates
     #3. Take downsampled (size=rf_size) image and coordinates
     #4. Whiten
     ###Parameters
    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    ##ImageSize - size of the translated image, from each we latter take chunk with size rf_size
    #TrLim - Maximal Amplitude of translation
    #smoothing - decides whether or not make texture correlated
    #image_whiten - determines whether separate textures should be whiten
    #set_whiten - determines whether the entire stimuli set should be whiten 
    from scipy.ndimage import gaussian_filter
    Image_Stream=[]
    Magnitude=[]
    
    Image=L
    ###smoothing
   
  ###Starting coordinates
    center_0=int(np.round((Image.shape[0]-rf_size*UpSample)/2))
    center_1=int(np.round((Image.shape[1]-rf_size*UpSample)/2))
    Coord=np.array((center_0,center_1))
    Image_Stream.append(Image[center_0:center_0+rf_size*UpSample:UpSample,
            center_1:center_1+rf_size*UpSample:UpSample])
    ###Determine translation steps
    for i in range (NumEx):

        it=np.random.randint(low=0,high=2)
        if it==0:
            V=np.random.randint(low=-TrLim,high=TrLim+1)   ###vertical shift
            H=0
        elif it==1:
            V=0
            H=np.random.randint(low=-TrLim,high=TrLim+1)

        Coord=Coord+(H,V)
       ##Samle image
        Image_T=Image[Coord[0]:Coord[0]+UpSample*rf_size:UpSample,
        Coord[1]:Coord[1]+UpSample*rf_size:UpSample]
        
        if image_whiten==1:

            Image_T=zca_whiten(Image_T)


        Image_Stream.append(Image_T)
        Magnitude.append([V,H])
    
    Image_Stream=np.array(Image_Stream)
    Magnitude=np.array(Magnitude)

    Image_Stream=np.reshape(Image_Stream,[NumEx+1,rf_size**2])
    
    if set_whiten==1:
        Image_Stream=zca_whiten(Image_Stream)

    Image_Stream=np.reshape(Image_Stream,[NumEx+1,rf_size**2,1])  

    return Image_Stream,Magnitude

###Translations of events
def Event_Simulator(Image_Stream,event_trh,Magnitude):

    ###event_trh- treshshold for generating an event
    Image_Stream=Image_Stream*20
    Image_Stream_temp=Image_Stream-np.amin(Image_Stream)+1

    Detector_Voltage=np.log(Image_Stream_temp)

    Event_Stream=np.zeros((Image_Stream.shape[0]-1,Image_Stream.shape[1]))

    Voltage_base=Detector_Voltage[0]
    for i in range(1,Detector_Voltage.shape[0]):
        for j in range(Image_Stream.shape[1]):
            if Detector_Voltage[i][j]-Voltage_base[j]>event_trh:
                Event_Stream[i-1][j]=1
                Voltage_base[j]=Detector_Voltage[i][j]
            elif Detector_Voltage[i][j]-Voltage_base[j]<-event_trh:
                Event_Stream[i-1][j]=-1
                Voltage_base[j]=Detector_Voltage[i][j]
    
    Magnitude=Magnitude[1:]

    return Detector_Voltage,Event_Stream,Magnitude

###Rotation  Generators 

###Rotation Generator where center of translation is not necessarily part of the receptive field:
def Rotation_Generator_Non_Center(NumEx,mu,sigma,rf_size,ImageSize,Rot_center,RotLim, smooth_std, smoothing=1,image_whiten=0,set_whiten=1):
    ###The idea is:
    ##1. Generate large correlated image
    ##2. Rotated around rotation_center
    ##3. Randomly generate coordinates for the receptive field center
    ##4. Sample image with receptive field size=rf_size from original and rotated big images 
    ##5. Whiten
    ###Parameters
    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    ###ImageSize=size of big image
    ##smooth_std - const for gaussian filter
    #Rot_center - center of rotation
    #RotLim - Maximal Angle of rotation

    from scipy.ndimage import gaussian_filter

    Original=[]
    Rotated=[]
    Magnitude=[]

    Image=np.random.normal(mu,sigma,ImageSize)

    if smoothing==1:
       Image=gaussian_filter(Image,smooth_std)
       Image=Image-np.mean(Image)
       Image=Image/np.std(Image)

###rotate
    for i in range (NumEx):
        A=np.random.randint(low=-RotLim,high=RotLim+1,size=1)
        A=int(A)
        RotM=cv2.getRotationMatrix2D(center=Rot_center,angle=A,scale=1)

        ImageR=cv2.warpAffine(Image,RotM,dsize=(ImageSize))
        ###Cut to avoid edge-artifacts
        ImageR=ImageR[int(0.1*ImageSize[0]):int(0.9*ImageSize[0]),int(0.1*ImageSize[0]):int(0.9*ImageSize[0])]

        edge_y=np.random.randint(0,ImageR.shape[0]-rf_size)
        edge_x=np.random.randint(0,ImageR.shape[0]-rf_size)
        ###Get "small" image
        ImageO=Image[int(0.1*ImageSize[0])+edge_y:int(0.1*ImageSize[0])+edge_y+rf_size,
                    int(0.1*ImageSize[0])+edge_x:int(0.1*ImageSize[0])+edge_x+rf_size]
        ImageR=ImageR[edge_y:edge_y+rf_size,edge_x:edge_x+rf_size]
        
        if image_whiten==1:
            ImageO=zca_whiten(ImageO)
            ImageR=zca_whiten(ImageR)

        Original.append(ImageO)
        Rotated.append(ImageR)
        Magnitude.append(A)

    Original=np.array(Original)
    Rotated=np.array(Rotated)

    if set_whiten==1:
        Original=np.reshape(Original,[NumEx,rf_size**2])
        Original=zca_whiten(Original)

        Rotated=np.reshape(Rotated,[NumEx,rf_size**2])
        Rotated=zca_whiten(Rotated)
    
    Original=np.reshape(Original,[NumEx,rf_size**2,1])
    Rotated=np.reshape(Rotated,[NumEx,rf_size**2,1])

    ImagePair=np.concatenate([Original,Rotated],axis=2)

    return ImagePair, Magnitude

###Rotation Generator where center of translation coincides with receptive_field center:
def Rotation_Generator_Center(NumEx,mu,sigma,rf_size,ImageSize,Rot_center,RotLim, smooth_std, smoothing=1,image_whiten=0,set_whiten=1):
    ###The plan is:
    #1. Generate correlated image
    #2. Rotate it
    #3. Sample from center with rf_size
    #4. If necessary - whiten

    ###Parameters
    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    #Rot_center - center of rotation
    #RotLim - Maximal Angle of rotation
    ###smothing and whitewning cosnstats
    from scipy.ndimage import gaussian_filter

    Original=[]
    Rotated=[]
    Magnitude=[]

    delta=int(rf_size/2)
    Coord_1=Rot_center[0]-delta
    Coord_2=Rot_center[0]+delta

    for i in range (NumEx):
        Image=np.random.normal(mu,sigma,ImageSize)
        if smoothing==1:
            Image=gaussian_filter(Image,smooth_std)
            Image=Image-np.mean(Image)
            Image=Image/np.std(Image)

        A=np.random.randint(low=-RotLim,high=RotLim+1,size=1)
        A=int(A)
        RotM=cv2.getRotationMatrix2D(center=Rot_center,angle=A,scale=1)

        ImageR=cv2.warpAffine(Image,RotM,dsize=(ImageSize))
        ImageR=ImageR[Coord_1:Coord_2,Coord_1:Coord_2]
        ImageO=Image[Coord_1:Coord_2,Coord_1:Coord_2]

        if image_whiten==1:
            ImageO=zca_whiten(ImageO)
            ImageR=zca_whiten(ImageR)

        Original.append(ImageO)
        Rotated.append(ImageR)
        Magnitude.append(A)

    Original=np.array(Original)
    Rotated=np.array(Rotated)

    if set_whiten==1:
        Original=np.reshape(Original,[NumEx,rf_size**2])
        Original=zca_whiten(Original)

        Rotated=np.reshape(Rotated,[NumEx,rf_size**2])
        Rotated=zca_whiten(Rotated)
    
    Original=np.reshape(Original,[NumEx,rf_size**2,1])
    Rotated=np.reshape(Rotated,[NumEx,rf_size**2,1])

    ImagePair=np.concatenate([Original,Rotated],axis=2)

    return ImagePair, Magnitude

###Rotations in the big Correlated World
def Rotation_Generator_World(NumEx,mu,sigma,rf_size,ImageSize,Rot_center,RotLim, smooth_std, smoothing=1,image_whiten=0,set_whiten=1):
    ###The idea is simple:
    #1. Create large correlated world
    #2. Randomly take small image from it
    #3. Rotate
    #4. Cut from that image subimgae with size=rf_size centered around center of rotation
    ###(This is to avoid edge-related artifacts during rotations)
    #5. Whiten
    
    ###Parameters
    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    #Rot_center - center of rotation
    #RotLim - Maximal Angle of rotation
    from scipy.ndimage import gaussian_filter

    Original=[]
    Rotated=[]
    Magnitude=[]

    Image=np.random.normal(mu,sigma,ImageSize)
    

    if smoothing==1:
       Image=gaussian_filter(Image,smooth_std)
       Image=Image-np.mean(Image)
       Image=Image/np.std(Image)

    for i in range (NumEx):

        rf_center=np.random.randint(rf_size,ImageSize-rf_size-1,2)
        ImageO=Image[rf_center[0]-rf_size:rf_center[0]+rf_size+1,rf_center[1]-rf_size:rf_center[1]+rf_size+1]
        
        A=np.random.randint(low=-RotLim,high=RotLim+1,size=1)
        A=int(A)
        RotM=cv2.getRotationMatrix2D(center=Rot_center,angle=A,scale=1)
        ImageR=cv2.warpAffine(ImageO,RotM,ImageO.shape)
        
        delta=int((ImageO.shape[0]-rf_size)/2)
    

        ImageO=ImageO[delta:-delta,delta:-delta]
        ImageR=ImageR[delta:-delta,delta:-delta]
        
        if image_whiten==1:
            ImageO=zca_whiten(ImageO)
            ImageR=zca_whiten(ImageR)

        Original.append(ImageO)
        Rotated.append(ImageR)
        Magnitude.append(A)

    Original=np.array(Original)
    Rotated=np.array(Rotated)

    if set_whiten==1:
        Original=np.reshape(Original,[NumEx,rf_size**2])
        Original=zca_whiten(Original)

        Rotated=np.reshape(Rotated,[NumEx,rf_size**2])
        Rotated=zca_whiten(Rotated)
    
    Original=np.reshape(Original,[NumEx,rf_size**2,1])
    Rotated=np.reshape(Rotated,[NumEx,rf_size**2,1])

    ImagePair=np.concatenate([Original,Rotated],axis=2)

    return ImagePair, Magnitude

###Functions related to model training
#Update Weight:
def UpdateWeight(W,Thetta,Thetta_hat,M,Delta,Xt,NChan,DVal,rf_size):
    #Thetta - output of network
    #Thetta_hat - sum of total neuronal activity through entire training
    #NChan - number of directions/unique neurons
    #DVal - Dimension i.e. 1D or 2 D
    dW=np.zeros((rf_size**DVal,rf_size**DVal,NChan))
    dM=np.zeros((NChan,NChan))
    for i in range(NChan):
        dW[:,:,i]=Thetta[i]*(np.outer(Delta,Xt.transpose())-W[:,:,i]*Thetta[i])/Thetta_hat[i]
    W+=dW
    for i in range(NChan):
        for j in range(NChan):
            dM[i,j]=Thetta[i]*(Thetta[j]-M[i,j]*Thetta[i])/Thetta_hat[i]
    np.fill_diagonal(dM,0.)
    M+=dM
    return (W,M)

###Train Model
def Train_Model (ImagePair,Magnitude,rf_size,NumEx,NChan,DVal,reweight,N_Epoch,Round_factor):
###IMPORTANT! Somehow learning is not stable,as it often the case with Hebbian plasticity.
###Some initializations lead to no learning at all, some explode coefficients.
###However,overall it works fine. So if result is weird, or unsatisfactory - run the function again.

    #Magnitude - Magnitudes of the stimuli translations
    #NumEx - number of training examples
    #NChan - number of directions
    #DVal - image dimension (i.e. 2D or 1D)
    #reweight - constant for accumulation of neuronal activity
    #Round_factor - techical parameter for sucessful usage WT and MT

    #t=ImagePair.shape
    #t=t[1]
    #Initialize
    Thetta_hat=1000*np.ones((NChan,1))
    Thetta=[]
    W=np.random.normal(0,1,[rf_size**DVal,rf_size**DVal,NChan])
    M=np.random.normal(0,1,[NChan,NChan])
    np.fill_diagonal(M,0.)

    ###
    Delta=ImagePair[:,:,1]-ImagePair[:,:,0]
    Delta=np.reshape(Delta,[NumEx,1,rf_size**DVal])  #temporal derivative
    Xt=ImagePair[:,:,0]                              #light intensity
    Xt=np.reshape(Xt,[NumEx,rf_size**DVal,1])
    Magnitude_training=[]
    #Track changes in M and W
    WT=np.zeros((int((NumEx+Round_factor)/100),rf_size**DVal,rf_size**DVal,NChan))
    MT=np.zeros((int((NumEx+Round_factor)/100),NChan,NChan))
    #Train
    for ll in range(N_Epoch):
        V=np.random.permutation(NumEx)
        for i in range(NumEx):
            if i%100 == 0:
                WT[int(0.01*i),:,:]=W
                MT[int(0.01*i),:,]=M
            Elayer=np.zeros((rf_size**DVal,NChan,1))  #weighting of pixel intenstisties  
            for j in range(NChan):
                Elayer[:,j]=np.dot(W[:,:,j],Xt[V[i]])
            Elayer=Elayer.reshape([rf_size**DVal,NChan])
            Elayer=np.dot(Delta[V[i]],Elayer).transpose() #Outer Product feature
            Tet=np.zeros((NChan,1))
            for k in range(200):     #Neuronal output
                

                Tet=relu(Elayer-np.dot(M,Tet))
            #Alternative approach to initialise weights
            #l2_x=np.linalg.norm(np.outer(Delta[V[i]],Xt[V[i]]),ord=2)**2
            #l2_y=np.linalg.norm(Tet,ord=2)**2
            #it=i
            #lambda_regul=10e-6
            #while (l2_x-l2_y)**2>lambda_regul and it<2:
            #       kk=it
            #       Tet[kk]=math.sqrt(abs(l2_x - np.linalg.norm(Tet,ord=2)**2))
            #       l2_y=np.linalg.norm(Tet,ord=2)**2
                   #it+=1
            Magnitude_training.append(Magnitude[V[i]])
            Thetta_hat+=reweight*(Tet*Tet)
            Thetta.append(Tet)
           # kk=0
        

            W,M=UpdateWeight(W,Tet,Thetta_hat,M,Delta[V[i]],Xt[V[i]],NChan,DVal,rf_size)

    Magnitude_training=np.array(Magnitude_training)
    if NChan==2:
        Magnitude_training=np.reshape(Magnitude_training,[Magnitude_training.shape[0],1])
    Thetta=np.array(Thetta)        
    Thetta=np.reshape(Thetta,[NumEx*N_Epoch,NChan]) 
    return W,M,Thetta,Thetta_hat,Delta,Xt,WT,MT,Magnitude_training

###Train Model in case the stimulus is a stream of images
def Train_Model_Stream (NumEx,mu,sigma,ImageSize,smooth_std,rf_size,UpSample,TrLim, NChan,DVal,reweight,N_Epoch):
###Slight adjustment of aforementioned script for the case of continuous images.
###The main difference are:
###Delta Xt are calculated on each step.
###For each epoch you generate "new" world where motion occurs. This is necessary, to avoid situation where
###as a result of movememnt coordinates of receptive field center get out of the "World"
    #Magnitude - Magnitudes of the stimuli translations
    #NumEx - number of training examples
    #NChan - number of directions
    #DVal - image dimension (i.e. 2D or 1D)
    #reweight - constant for accumulation of neuronal activity


    #t=ImagePair.shape
    #t=t[1]
    #Initialize
    Thetta_hat=1000*np.ones((NChan,1))
    Thetta=[]
    W=np.random.normal(0,1,[rf_size**DVal,rf_size**DVal,NChan])
    M=np.random.normal(0,1,[NChan,NChan])
    np.fill_diagonal(M,0.)

    ###
   
    Magnitude_training=[]
    #Track changes in M and W
    WT=[]
    MT=[]
    #Train
    for ll in range(N_Epoch):
        Image_Stream,Magnitude=Translation_Generator_2D_Stream(NumEx,mu,sigma,ImageSize,smooth_std,rf_size,UpSample,TrLim,smoothing=1,image_whiten=0,set_whiten=1)
        
        for i in range(1,NumEx+1):

            Xt=Image_Stream[i-1]
            Delta=Image_Stream[i]-Image_Stream[i-1]
            Delta=Delta.T

            Elayer=np.zeros((rf_size**DVal,NChan,1))  #weighting of pixel intenstisties  
            for j in range(NChan):
                Elayer[:,j]=np.dot(W[:,:,j],Xt)
            Elayer=Elayer.reshape([rf_size**DVal,NChan])
            Elayer=np.dot(Delta,Elayer).transpose() #Outer Product feature
            Tet=np.zeros((NChan,1))

            for k in range(200):     #Neuronal output
                Tet=relu(Elayer-np.dot(M,Tet))

            Magnitude_training.append(Magnitude[i-1])
            Thetta_hat+=reweight*(Tet*Tet)
            Thetta.append(Tet)

            W,M=UpdateWeight(W,Tet,Thetta_hat,M,Delta,Xt,NChan,DVal,rf_size)

            if i%100==0:
                WT.append(W)
                MT.append(M)
    Magnitude_training=np.array(Magnitude_training)    
    if NChan==2:
        Magnitude_training=np.reshape(Magnitude_training,[Magnitude_training.shape[0],1])
    Thetta=np.array(Thetta)        
    Thetta=np.reshape(Thetta,[NumEx*N_Epoch,NChan]) 
    return W,M,Thetta,Thetta_hat,WT,MT,Magnitude_training

def Train_Model_Stream_Event(NumEx,mu,sigma,ImageSize,smooth_std,rf_size,UpSample,event_trh,TrLim, NChan,DVal,reweight,N_Epoch,event_whiten=0,EPS=1e-2):
###Slight adjustment of aforementioned script for the case of continuous images.
###The main difference are:
###Delta Xt are calculated on each step.
###For each epoch you generate "new" world where motion occurs. This is necessary, to avoid situation where
###as a result of movememnt coordinates of receptive field center get out of the "World"
    #Magnitude - Magnitudes of the stimuli translations
    #NumEx - number of training examples
    #NChan - number of directions
    #DVal - image dimension (i.e. 2D or 1D)
    #reweight - constant for accumulation of neuronal activity


    #t=ImagePair.shape
    #t=t[1]
    #Initialize
    Thetta_hat=1000*np.ones((NChan,1))
    Thetta=[]
    W=np.random.normal(0,0,[rf_size**DVal,rf_size**DVal,NChan])
    M=np.random.normal(0,0,[NChan,NChan])
    np.fill_diagonal(M,0.)

    ###
   
    Magnitude_training=[]
    #Track changes in M and W
    WT=[]
    MT=[]
    #Train
    for ll in range(N_Epoch):
        Image_Stream,Magnitude=Translation_Generator_2D_Stream(NumEx,mu,sigma,ImageSize,smooth_std,rf_size,UpSample,TrLim,smoothing=1,image_whiten=0,set_whiten=0)
        Detector_Voltage,Event_Stream,Magnitude=Event_Simulator(Image_Stream,event_trh,Magnitude)
        Event_Stream=np.reshape(Event_Stream,[NumEx,rf_size**2])
        if event_whiten==1:
            Event_Stream=zca_whiten(Event_Stream,EPS)
        Event_Stream=np.reshape(Event_Stream,[NumEx,rf_size**2,1])
        for i in range(1,NumEx):

            Xt=Event_Stream[i-1]
            Delta=Event_Stream[i]-Event_Stream[i-1]
            Delta=Delta.T

            Elayer=np.zeros((rf_size**DVal,NChan,1))  #weighting of pixel intenstisties  
            for j in range(NChan):
                Elayer[:,j]=np.dot(W[:,:,j],Xt)
            Elayer=Elayer.reshape([rf_size**DVal,NChan])
            Elayer=np.dot(Delta,Elayer).transpose() #Outer Product feature
            Tet=np.zeros((NChan,1))

            for k in range(200):     #Neuronal output
                Tet=relu(Elayer-np.dot(M,Tet))

            Magnitude_training.append(Magnitude[i-1])
            Thetta_hat+=reweight*(Tet*Tet)
            Thetta.append(Tet)

            W,M=UpdateWeight(W,Tet,Thetta_hat,M,Delta,Xt,NChan,DVal,rf_size)

            if i%100==0:
                WT.append(W)
                MT.append(M)
    Magnitude_training=np.array(Magnitude_training)    
    if NChan==2:
        Magnitude_training=np.reshape(Magnitude_training,[Magnitude_training.shape[0],1])
    Thetta=np.array(Thetta)        
    Thetta=np.reshape(Thetta,[(NumEx-1)*N_Epoch,NChan]) 


                #it=i
            #lambda_regul=10e-6
            #while (l2_x-l2_y)**2>lambda_regul and it<2:
            #       kk=it
            #       Tet[kk]=math.sqrt(abs(l2_x - np.linalg.norm(Tet,ord=2)**2))
            #       l2_y=np.linalg.norm(Tet,ord=2)**2
                   #it+=1
    return W,M,Thetta,Thetta_hat,WT,MT,Magnitude_training

def Train_Model_Stream_Event_2(NumEx,mu,sigma,ImageSize,smooth_std,rf_size,UpSample,event_trh,TrLim, NChan,DVal,reweight,N_Epoch,event_whiten=0,EPS=1e-2):
###Slight adjustment of aforementioned script for the case of continuous images.
###The main difference are:
###Delta Xt are calculated on each step.
###For each epoch you generate "new" world where motion occurs. This is necessary, to avoid situation where
###as a result of movememnt coordinates of receptive field center get out of the "World"
    #Magnitude - Magnitudes of the stimuli translations
    #NumEx - number of training examples
    #NChan - number of directions
    #DVal - image dimension (i.e. 2D or 1D)
    #reweight - constant for accumulation of neuronal activity


    #t=ImagePair.shape
    #t=t[1]
    #Initialize
    Thetta_hat=1000*np.ones((NChan,1))
    Thetta=[]
    W=np.random.normal(0,0,[rf_size**DVal,rf_size**DVal,NChan])
    M=np.random.normal(0,0,[NChan,NChan])
    np.fill_diagonal(M,0.)

    ###
   
    Magnitude_training=[]
    #Track changes in M and W
    WT=[]
    MT=[]
    #Train
    for ll in range(N_Epoch):
        Image_Stream,Magnitude=Translation_Generator_2D_Stream(NumEx,mu,sigma,ImageSize,smooth_std,rf_size,UpSample,TrLim,smoothing=1,image_whiten=0,set_whiten=0)
        Detector_Voltage,Event_Stream,Magnitude=Event_Simulator(Image_Stream,event_trh,Magnitude)
        Event_Stream=np.reshape(Event_Stream,[NumEx,rf_size**2])
        if event_whiten==1:
            Event_Stream=zca_whiten(Event_Stream,EPS)
        Event_Stream=np.reshape(Event_Stream,[NumEx,rf_size**2,1])
        for i in range(1,NumEx):

            Xt=Event_Stream[i-1]
            Delta=Event_Stream[i]-Event_Stream[i-1]
            Delta=Delta.T

            Elayer=np.zeros((rf_size**DVal,NChan,1))  #weighting of pixel intenstisties  
            for j in range(NChan):
                Elayer[:,j]=np.dot(W[:,:,j],Xt)
            Elayer=Elayer.reshape([rf_size**DVal,NChan])
            Elayer=np.dot(Delta,Elayer).transpose() #Outer Product feature
            Tet=np.zeros((NChan,1))

            for k in range(200):     #Neuronal output
                Tet=relu(Elayer-np.dot(M,Tet))
            
            l2_x=np.linalg.norm(np.outer(Delta,Xt),ord=2)**2
            l2_y=np.linalg.norm(Tet,ord=2)**2
            it=i-1
            lambda_regul=1e-6
            while ll==0 and (l2_x-l2_y)**2>lambda_regul and it<4:
                kk=it
                Tet[kk]=math.sqrt(abs(l2_x - np.linalg.norm(Tet,ord=2)**2))
                l2_y=np.linalg.norm(Tet,ord=2)**2
                it+=1

            Magnitude_training.append(Magnitude[i-1])
            Thetta_hat+=reweight*(Tet*Tet)
            Thetta.append(Tet)

            W,M=UpdateWeight(W,Tet,Thetta_hat,M,Delta,Xt,NChan,DVal,rf_size)

            if i%100==0:
                WT.append(W)
                MT.append(M)
    Magnitude_training=np.array(Magnitude_training)    
    if NChan==2:
        Magnitude_training=np.reshape(Magnitude_training,[Magnitude_training.shape[0],1])
    Thetta=np.array(Thetta)        
    Thetta=np.reshape(Thetta,[(NumEx-1)*N_Epoch,NChan]) 


    return W,M,Thetta,Thetta_hat,WT,MT,Magnitude_training
###Scripts to Asses quality of training|performance
#Training quality
def Training_Quality(Magnitude,Thetta,test_size):
###Simple script which:
# 1. takes transformation amplitudes of the last test_size stimulus iterations
# 2. correlates them with last test_size steps of detector outputs.
#3. Assigns detectors to directions they encode.
#4. Subtract outputs of the detectors which encode opponent motion directions to yield how good motion along an axis is encode.
###Plus, it is also kinda bioinspired
###Take last test_size magnitudes and detector outpysus
    Mag_test=Magnitude[-test_size:]
    Thetta_test=Thetta[-test_size:]
###Determine which director belongs to which directions
    Corrs=np.zeros((Mag_test.shape[1],Thetta_test.shape[1]))
    for i in range(Mag_test.shape[1]):
        for j in range(Thetta_test.shape[1]):
            Corrs[i,j]=np.amin(np.corrcoef(Mag_test[:,i],Thetta_test[:,j]))
    if Mag_test.shape[1]==2:
        Directions=['Vertical+','Vertical-','Horizontal+','Horizontal-']
    elif Mag_test.shape[1]==1:
        Directions=['Vertical+','Verical-']

    detector_ind=[]
    Dir_Corr=[]
    Axis_Corr=[]
    for k in range(Mag_test.shape[1]):
       detector_ind.append(np.where(Corrs[k,:]==np.amax(Corrs[k,:]))[0][0])
       detector_ind.append(np.where(Corrs[k,:]==np.amin(Corrs[k,:]))[0][0])

       Dir_Corr.append(np.amin(np.corrcoef(Mag_test[:,k][Mag_test[:,k]>0],
       Thetta_test[:,detector_ind[2*k]][Mag_test[:,k]>0])))
       Dir_Corr.append(np.amin(np.corrcoef(Mag_test[:,k][Mag_test[:,k]<0],
       Thetta_test[:,detector_ind[2*k+1]][Mag_test[:,k]<0])))
       
       Axis_Corr.append(np.amin(np.corrcoef(Mag_test[:,k][Mag_test[:,k]!=0],
       Thetta_test[:,detector_ind[2*k]][Mag_test[:,k]!=0]-Thetta_test[:,detector_ind[2*k+1]][Mag_test[:,k]!=0])))
       Axis_Corr.append('')

    Detector_Asses=pd.DataFrame([detector_ind,Dir_Corr,Axis_Corr],index=['Position','Correlation_Detector','Correlation_Axis'],columns=Directions)
    
    return Detector_Asses,Mag_test,Thetta_test

###Model Evaluation
def Eval_Model (ImagePair,Magnitude,rf_size,NumEx,NChan,DVal,N_Epoch,W,M,det_as):
    ###Sctript to estimate how good detectors respond to motions from another dataset/superpixel motions
    ###The idea is:
    #1. Essentially to repeat train_model function, but without weight update
    #2. Subtract output of detectors with opposite directions to determine how good an axis is encode 
    #Magnitude - Magnitudes of the stimuli translations
    #NumEx - number of training examples
    #NChan - number of directions
    #DVal - image dimension (i.e. 2D or 1D)
    #reweight - constant for accumulation of neuronal activity
    #Round_factor - techical parameter for sucessful usage WT and MT
    #W         - excitatory weights
    #M  - inhibitory weights
    #det_ass information about directions and detectors
    #Return - output along the axis, transformation magnitudes

    det_ind=np.array(det_as.loc['Position'].astype(int))

    ###
    Delta=ImagePair[:,:,1]-ImagePair[:,:,0]
    Delta=np.reshape(Delta,[NumEx,1,rf_size**DVal])  #temporal derivative
    Xt=ImagePair[:,:,0]                              #light intensity
    Xt=np.reshape(Xt,[NumEx,rf_size**DVal,1])
    Magnitude_training=[]
    Thetta_Net=[]

    #Evaluate
    for ll in range(N_Epoch):
        V=np.random.permutation(NumEx)
        for i in range(NumEx):
            Elayer=np.zeros((rf_size**DVal,NChan,1))  #weighting of pixel intenstisties  
            for j in range(NChan):
                Elayer[:,j]=np.dot(W[:,:,j],Xt[V[i]])
            Elayer=Elayer.reshape([rf_size**DVal,NChan])
            Elayer=np.dot(Delta[V[i]],Elayer).transpose() #Outer Product feature
            Tet=np.zeros((NChan,1))
            for k in range(200):     #Neuronal output
                Tet=relu(Elayer-np.dot(M,Tet))
            Tet_ax=np.zeros(int(NChan/2))
            for t in range(int(NChan/2)):
                Tet_ax[t]=Tet[det_ind[2*t]]-Tet[det_ind[2*t+1]]
            
            Magnitude_training.append(Magnitude[V[i]])
            Thetta_Net.append(Tet_ax)        
    Thetta_Net=np.array(Thetta_Net)   
    Magnitude_training=np.array(Magnitude_training) 
    if NChan==2:
        Magnitude_training=np.reshape(Magnitude_training,[Magnitude_training.shape[0],1])
    return Thetta_Net,Magnitude_training

###Determine how good model does with validation dataset
def Model_Validation (Thetta_Net,Magnitude_training_large):
###The goal is to asses quality of responses to validation data set.
####1. Get all of the unique magnitudes for each of the axis
####2. Get responses to the unique magnitudes
###3. Average responses to each magnitude. Determine standard deviaion and r.m.s. - the relation between std and mean 
###for eah magnitude
    UniqueMag=[]
    for i in range (Thetta_Net.shape[1]):
        UniqueMag.append(np.unique(Magnitude_training_large[:,i]))

    UniqueMag=np.array(UniqueMag)
    UniqueMag=UniqueMag.T

    Index=[]
    for j in range (Thetta_Net.shape[1]):
        for k in range (len(UniqueMag)):
            searchval=UniqueMag[k,j]
            ii=np.where(Magnitude_training_large==searchval)[0]
            Index.append(ii)
    Response_Mean=[]
    Response_Dev=[]
    Response_Error=[]
    for ll in range(Thetta_Net.shape[1]):


        R_Mean_temp=[]
        R_Dev_temp=[]
        R_Error_temp=[]
        for t in range(len(UniqueMag)):

            temp=Thetta_Net[Index[len(UniqueMag)*ll+t],ll]
            Mean_temp=np.mean(temp)
            Dev_temp=np.std(temp)
            Error_temp=Dev_temp/Mean_temp
            R_Mean_temp.append(Mean_temp)
            R_Dev_temp.append(Dev_temp)
            R_Error_temp.append(Error_temp)
        Response_Mean.append(R_Mean_temp)
        Response_Dev.append(R_Dev_temp)
        Response_Error.append(R_Error_temp)

    Response_Mean=np.array(Response_Mean)
    Response_Mean=Response_Mean.T

    Response_Dev=np.array(Response_Dev)
    Response_Dev=Response_Dev.T

    Response_Error=np.array(Response_Error)
    Response_Error=Response_Error.T
    
    return(Response_Mean,Response_Error,Response_Dev,UniqueMag)

###Visualisation funcions

###Visualisation of the training performance
def Performance_Visualisation_Train(det_as,Mag_test,Thetta_test,limit):
    ###It plots together outputs along the axis for the stimulus iterations determined by limit
    for i in range(Mag_test.shape[1]):
        plt.figure()
        it1=det_as.loc['Position'][2*i].astype(int)
        it2=det_as.loc['Position'][2*i+1].astype(int)
        plt.plot(Mag_test[limit[0]:limit[1],i])
        plt.plot(Thetta_test[limit[0]:limit[1],it1]-Thetta_test[limit[0]:limit[1],it2])

###Visualisation of the Validation performance
def Performance_Visualisation_Validation(UniqueMag,Response_Mean,Response_Dev):
    ###Plot determined by validation function mean and std for each stimulus magnitude versus stimulus magnitude.
    
    for i in range(UniqueMag.shape[1]):
        plt.figure()
        plt.plot(UniqueMag[:,i]/10,Response_Mean[:,i],'g-')
        plt.fill_between(UniqueMag[:,i]/10,Response_Mean[:,i]-Response_Dev[:,i],Response_Mean[:,i]+Response_Dev[:,i],alpha=0.2)
        plt.show()

###Visualisation of detector weights learned in 2d world
# ###For the 1 d it is trivial            
def Detector_2D_Visualisation(Number,rf_size,W):
##While for the 1d case weight arrangement is trivial, here one needs some fiddling
####Essentially we reshape weight matrix with size (rf_size**2,rf_size**2) 
###into rf_size**2 matrixes with dimensions rf_size,rf_size
###each rf_size,rf_size matrix is the weights for the image within receptive field for the
###each value of Delta. So you range in it rf_size by rf_size grid of rf_size by rf_size matrixes,
###where position whithin a grid, corresponds to the pixel in Detla
   fig, axes = plt.subplots(rf_size, rf_size, figsize=(rf_size**2, rf_size**2))
   for k in range(Number,Number+1):
        fig.suptitle( 'DetectorNo'+str(k+1))
        for i in range(rf_size):
            for j in range(rf_size):
                sns.heatmap(np.reshape(W[rf_size*i+j,:,k],(rf_size,rf_size)),cmap="gray",ax=axes[i,j])
def Translation_Generator_2D_Stream_3(NumEx,mu,sigma,ImageSize,smooth_traj,smooth_std,rf_size,UpSample,smoothing=1,image_whiten=1,set_whiten=1):
     ##Here we augmented original paper by Bahroun et al. by adding "stream of images" as a stimuli.
     ##In the original paper stimuli at each training step consisted of pair image-translated image.
     ###Here we generate sequence of images and train our network to take an image at time t, compare it 
     ### with image at moment t-1 and find transformation operator.
     ###At the moment it does only vertical or horizontal at each step, but it can be easily changed to simultaneous translations
     ###So the idea is:
     #1. Create correlated, superpixel 2D world
     #2. Update coordinates
     #3. Take downsampled (size=rf_size) image and coordinates
     #4. Whiten
     ###Parameters
    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    ##ImageSize - size of the translated image, from each we latter take chunk with size rf_size
    #TrLim - Maximal Amplitude of translation
    #smoothing - decides whether or not make texture correlated
    #image_whiten - determines whether separate textures should be whiten
    #set_whiten - determines whether the entire stimuli set should be whiten 
    from scipy.ndimage import gaussian_filter
    
    trajectory=np.random.normal(mu,sigma,[NumEx,1])
    trajectory=gaussian_filter(trajectory,smooth_traj)
    trajectory=trajectory-np.mean(trajectory)
    #trajectory=trajectory/np.std(trajectory)
    trajectory=trajectory*5
    trajectory=trajectory.astype(int)
     
    Image_Stream=[]

    
    Image=np.random.normal(mu,sigma,ImageSize)
    ###smoothing
    if smoothing==1:

        Image=gaussian_filter(Image,smooth_std)
        Image=Image-np.mean(Image)
        Image=Image/np.std(Image)
  ###Starting coordinates
    center=int(np.round((ImageSize[0]-rf_size*UpSample)/2))
    Coord=np.array((center,center))
    Image_Stream.append(Image[center:center+rf_size*UpSample:UpSample,
          center:center+rf_size*UpSample:UpSample])
    ###Determine translation steps
    for i in range (NumEx):

        tran=np.array((trajectory[i][0],0))
        Coord=Coord+tran
        Image_T=Image[Coord[0]:Coord[0]+UpSample*rf_size:UpSample,
        Coord[1]:Coord[1]+UpSample*rf_size:UpSample]

        
        if image_whiten==1:

            Image_T=zca_whiten(Image_T)


        Image_Stream.append(Image_T)
        
    
    Image_Stream=np.array(Image_Stream)
    Magnitude=trajectory

    Image_Stream=np.reshape(Image_Stream,[NumEx+1,rf_size**2])
    
    if set_whiten==1:
        Image_Stream=zca_whiten(Image_Stream)

    Image_Stream=np.reshape(Image_Stream,[NumEx+1,rf_size**2,1])  

    return Image_Stream,Magnitude

def Train_Model_Stream_3 (NumEx,mu,sigma,ImageSize,smooth_traj,smooth_std,rf_size,UpSample, NChan,DVal,reweight,N_Epoch):
###Slight adjustment of aforementioned script for the case of continuous images.
###The main difference are:
###Delta Xt are calculated on each step.
###For each epoch you generate "new" world where motion occurs. This is necessary, to avoid situation where
###as a result of movememnt coordinates of receptive field center get out of the "World"
    #Magnitude - Magnitudes of the stimuli translations
    #NumEx - number of training examples
    #NChan - number of directions
    #DVal - image dimension (i.e. 2D or 1D)
    #reweight - constant for accumulation of neuronal activity


    #t=ImagePair.shape
    #t=t[1]
    #Initialize
    Thetta_hat=1000*np.ones((NChan,1))
    Thetta=[]
    W=np.random.normal(0,1,[rf_size**DVal,rf_size**DVal,NChan])
    M=np.random.normal(0,1,[NChan,NChan])
    np.fill_diagonal(M,0.)

    ###
   
    Magnitude_training=[]
    #Track changes in M and W
    WT=[]
    MT=[]
    #Train
    for ll in range(N_Epoch):
        Image_Stream,Magnitude=Translation_Generator_2D_Stream_3(NumEx,mu,sigma,ImageSize,smooth_traj,smooth_std,rf_size,UpSample,smoothing=1,image_whiten=0,set_whiten=1)
        
        for i in range(1,NumEx+1):

            Xt=Image_Stream[i-1]
            Delta=Image_Stream[i]-Image_Stream[i-1]
            Delta=Delta.T

            Elayer=np.zeros((rf_size**DVal,NChan,1))  #weighting of pixel intenstisties  
            for j in range(NChan):
                Elayer[:,j]=np.dot(W[:,:,j],Xt)
            Elayer=Elayer.reshape([rf_size**DVal,NChan])
            Elayer=np.dot(Delta,Elayer).transpose() #Outer Product feature
            Tet=np.zeros((NChan,1))

            for k in range(200):     #Neuronal output
                Tet=relu(Elayer-np.dot(M,Tet))

            Magnitude_training.append(Magnitude[i-1])
            Thetta_hat+=reweight*(Tet*Tet)
            Thetta.append(Tet)

            W,M=UpdateWeight(W,Tet,Thetta_hat,M,Delta,Xt,NChan,DVal,rf_size)

            if i%100==0:
                WT.append(W)
                MT.append(M)
    Magnitude_training=np.array(Magnitude_training)    
    if NChan==2:
        Magnitude_training=np.reshape(Magnitude_training,[Magnitude_training.shape[0],1])
    Thetta=np.array(Thetta)        
    Thetta=np.reshape(Thetta,[NumEx*N_Epoch,NChan]) 
    return W,M,Thetta,Thetta_hat,WT,MT,Magnitude_training




def Translation_Generator_2D_Stream_Naturel(My_Image,NumEx,rf_size,DVal,UpSample,TrLim,image_whiten=0,set_whiten=0):

     ##Here we augmented original paper by Bahroun et al. by adding "stream of images" as a stimuli.
     ##In the original paper stimuli at each training step consisted of pair image-translated image.
     ###Here we generate sequence of images and train our network to take an image at time t, compare it 
     ### with image at moment t-1 and find transformation operator.
     ###At the moment it does only vertical or horizontal at each step, but it can be easily changed to simultaneous translations
     ###So the idea is:
     #1. Create correlated, superpixel 2D world
     #2. Update coordinates
     #3. Take downsampled (size=rf_size) image and coordinates
     #4. Whiten
     ###Parameters
    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    ##ImageSize - size of the translated image, from each we latter take chunk with size rf_size
    #TrLim - Maximal Amplitude of translation
    #smoothing - decides whether or not make texture correlated
    #image_whiten - determines whether separate textures should be whiten
    #set_whiten - determines whether the entire stimuli set should be whiten 
    from scipy.ndimage import gaussian_filter
    from PIL import Image
    Set_trans=np.arange(-TrLim,TrLim+1,1)
    Set_trans=Set_trans[Set_trans!=0]
    trajectory=np.random.choice(Set_trans,NumEx, replace=True)
    trajectory=trajectory.astype(int)
    #trajectory=trajectory/np.std(trajectory)

    Image_Stream=[]
    Magnitude=[]
 
    ImageSize=My_Image.shape
    ###smoothing
  ###Starting coordinates
    center=np.random.randint(250,np.amin(ImageSize)-250)
    Coord=np.array((center,center))
    Image_Stream.append(My_Image[center:center+rf_size*UpSample:UpSample,
            center:center+rf_size*UpSample:UpSample])
    ###Determine translation steps
    for i in range (NumEx):

        tran=np.array((trajectory[i],0))
        Coord=Coord+tran
        Image_T=My_Image[Coord[0]:Coord[0]+UpSample*rf_size:UpSample,
        Coord[1]:Coord[1]+UpSample*rf_size:UpSample]

        if image_whiten==1:

            Image_T=zca_whiten(Image_T)


        Image_Stream.append(Image_T)
        
    
    Image_Stream=np.array(Image_Stream)
    Magnitude=trajectory

    Image_Stream=np.reshape(Image_Stream,[NumEx+1,rf_size**DVal])
    
    if set_whiten==1:
        Image_Stream=zca_whiten(Image_Stream)

    Image_Stream=np.reshape(Image_Stream,[NumEx+1,rf_size**DVal,1])  

    return Image_Stream,Magnitude

def Train_Model_Stream_Naturel (NumEx,rf_size,UpSample,TrLim,NChan,DVal,reweight,N_Epoch,image_whiten=0,set_whiten=0,s_gauss=2,world_whiten=0):
###Slight adjustment of aforementioned script for the case of continuous images.
###The main difference are:
###Delta Xt are calculated on each step.
###For each epoch you generate "new" world where motion occurs. This is necessary, to avoid situation where
###as a result of movememnt coordinates of receptive field center get out of the "World"
    #Magnitude - Magnitudes of the stimuli translations
    #NumEx - number of training examples
    #NChan - number of directions
    #DVal - image dimension (i.e. 2D or 1D)
    #reweight - constant for accumulation of neuronal activity

    from PIL import Image
    from scipy.ndimage import gaussian_filter

    #t=ImagePair.shape
    #t=t[1]
    #Initialize
    Thetta_hat=10000*np.ones((NChan,1))
    Thetta=[]
    W=np.random.normal(0,1,[rf_size**DVal,rf_size**DVal,NChan])
    M=np.random.uniform(0.4,0.6,[NChan,NChan])
    np.fill_diagonal(M,0.)
    Delta_boss=[]
    Xt_boss=[]
    
    ###
   
    Magnitude_training=[]
    #Track changes in M and W
    Wt=np.zeros([N_Epoch,rf_size**DVal,rf_size**DVal,NChan])
    Mt=np.zeros([N_Epoch,NChan,NChan])
    Image_Set=[]
    for it in range(2,10):
        if it<10:
            it=str(0)+str(it)
        temp=Image.open(("/home/myedut/Downloads/Bahroun Replication/Naturalistic_Images/UPenn_Images/DSC_00"+str(it)+".JPG"))
        temp=np.array(temp)
        temp=temp[:,:,0]
        temp=temp-np.mean(temp)
        temp=temp/np.std(temp)
        Image_Set.append(temp)
    it=np.random.randint(0,9)
    My_Image=Image_Set[it]
    My_Image=gaussian_filter(My_Image,s_gauss)
    if world_whiten==1:
        My_Image=zca_whiten(My_Image)
    #Train
    for ll in range(N_Epoch):
        if DVal==2:

            Image_Stream,Magnitude=Translation_Generator_2D_Stream_Naturel(My_Image,NumEx,rf_size,DVal,UpSample,TrLim,image_whiten=image_whiten,set_whiten=set_whiten)
        elif DVal==1:
            Image_Stream,Magnitude=Translation_Generator_1D_Stream_Naturel(My_Image,NumEx,rf_size,DVal,UpSample,TrLim,image_whiten=image_whiten,set_whiten=set_whiten)

        for i in range(1,NumEx+1):
            
            Xt=Image_Stream[i-1]
            Delta=Image_Stream[i]-Image_Stream[i-1]
            Delta=Delta.T

            Elayer=np.zeros((rf_size**DVal,NChan,1))  #weighting of pixel intenstisties  
            for j in range(NChan):
                Elayer[:,j]=np.dot(W[:,:,j],Xt)
            Elayer=Elayer.reshape([rf_size**DVal,NChan])
            Elayer=np.dot(Delta,Elayer).transpose() #Outer Product feature
            Tet=np.zeros((NChan,1))

            for k in range(200):     #Neuronal output
                Tet=relu(Elayer-np.dot(M,Tet))

            Magnitude_training.append(Magnitude[i-1])
            Thetta_hat+=reweight*(Tet*Tet)
            Thetta.append(Tet)
            Delta_boss.append(Delta)
            Xt_boss.append(Xt)
            Wt[ll]=W
            Mt[ll]=M
            
            W,M=UpdateWeight(W,Tet,Thetta_hat,M,Delta,Xt,NChan,DVal,rf_size)

            
           
    Magnitude_training=np.array(Magnitude_training)    
    if NChan==2:
        Magnitude_training=np.reshape(Magnitude_training,[Magnitude_training.shape[0],1])
    Thetta=np.array(Thetta)        
    Thetta=np.reshape(Thetta,[NumEx*N_Epoch,NChan]) 
    return W,M,Thetta,Thetta_hat,Wt,Mt,Magnitude_training,Image_Stream,set_whiten,Delta_boss,Xt_boss

def Translation_Generator_1D_Stream_Naturel(My_Image,NumEx,rf_size,DVal,UpSample,TrLim,image_whiten=0,set_whiten=0):

     ##Here we augmented original paper by Bahroun et al. by adding "stream of images" as a stimuli.
     ##In the original paper stimuli at each training step consisted of pair image-translated image.
     ###Here we generate sequence of images and train our network to take an image at time t, compare it 
     ### with image at moment t-1 and find transformation operator.
     ###At the moment it does only vertical or horizontal at each step, but it can be easily changed to simultaneous translations
     ###So the idea is:
     #1. Create correlated, superpixel 2D world
     #2. Update coordinates
     #3. Take downsampled (size=rf_size) image and coordinates
     #4. Whiten
     ###Parameters
    #NumEx - number of training examples
    #mu - image mean
    #sigma - image standard deviation
    ##ImageSize - size of the translated image, from each we latter take chunk with size rf_size
    #TrLim - Maximal Amplitude of translation
    #smoothing - decides whether or not make texture correlated
    #image_whiten - determines whether separate textures should be whiten
    #set_whiten - determines whether the entire stimuli set should be whiten 
    from scipy.ndimage import gaussian_filter
    from PIL import Image
    Set_trans=np.arange(-TrLim,TrLim+1,1)
    Set_trans=Set_trans[Set_trans!=0]
    trajectory=np.random.choice(Set_trans,NumEx, replace=True)
    trajectory=trajectory.astype(int)
    #trajectory=trajectory/np.std(trajectory)

    Image_Stream=[]
    Magnitude=[]
 
    ImageSize=My_Image.shape
    ###smoothing
  ###Starting coordinates
    center=np.random.randint(500,np.amin(ImageSize)-500)
    Coord=np.array((center,center))
    Image_Stream.append(My_Image[center:center+rf_size*UpSample:UpSample,
            center])
    
    ###Determine translation steps
    for i in range (NumEx):

        tran=np.array((trajectory[i],0))
        Coord=Coord+tran
        Image_T=My_Image[Coord[0]:Coord[0]+UpSample*rf_size:UpSample,
        Coord[1]]

        if image_whiten==1:

            Image_T=zca_whiten(Image_T)


        Image_Stream.append(Image_T)
        
    
    Image_Stream=np.array(Image_Stream)
    Magnitude=trajectory

    Image_Stream=np.reshape(Image_Stream,[NumEx+1,rf_size**DVal])
    
    if set_whiten==1:
        Image_Stream=zca_whiten(Image_Stream)

    Image_Stream=np.reshape(Image_Stream,[NumEx+1,rf_size**DVal,1])  

    return Image_Stream,Magnitude
