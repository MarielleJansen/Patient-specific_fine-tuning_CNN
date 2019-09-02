# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:05:00 2018

@author: Mariëlle Jansen

Liver lesion segmentation with P-net DCEDWI
Supervised fine-tuning

"""
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dropout, BatchNormalization, Activation
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint 
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from os import path
import scipy.ndimage.interpolation as sp

def read_image(filename):
    nii = nib.load(filename)  # Use nibabel to load nifti image
    data = nii.get_data()
    data = np.asarray(data)
    #data = np.swapaxes(data, 0,1)
    return data   

def standardization(image):
    s1 = 0  # minimum value mapping
    s2 = 1  # maximum value mapping
    pc2 = 0.998 # maximum landmark 99.8th percentile
    
    X = np.ndarray.flatten(image)
    X_sorted = np.sort(X)
    
    p1 = 0 
    p2 = X_sorted[np.round(len(X)*pc2+1).astype(int)]
    st = (s2-s1)/(p2-p1)
    
    image_mapped = np.zeros(X.shape,dtype='float32')
    X.astype('float32')
    image_mapped = s1 + X*st
    
    image_mapped[np.where(image_mapped<0)] = 0
    image_mapped = image/100 # In case of already normalized images
    meanIm = np.mean(image_mapped)
    stdIm = np.std(image_mapped)
    
    Im = (image_mapped-meanIm)/stdIm    # zero mean unit variance for Neural network
    
    Im = np.reshape(Im, image.shape)
    Im.astype('float32')
    return Im

# Load DCE-MR images, DW-MR images, lesion mask, and liver mask
def load_data(basedir, nameImage, number):
    basedir = basedir
    # Load train data; find body mask and normalize
    mask = read_image(path.join(basedir, str(number),'3DLesionAnnotations.nii'))
    mask = np.swapaxes(mask, 0,2)
    liverMask = read_image(path.join(basedir, str(number),'LiverMask_dilated.nii'))
    liverMask = np.swapaxes(liverMask, 0,2)
    liverMask = liverMask+mask
    idx = liverMask > 0
    liverMask[idx] = 1
    
    image = read_image(path.join(basedir, str(number),nameImage))
    normImage = standardization(image)   
    
    Im1 = normImage[:,:,:,0]
    Im1 = Im1[:,:,:,np.newaxis]
    Im2 = np.mean(normImage[:,:,:,1:6], axis=3)
    Im2 = Im2[:,:,:,np.newaxis]
    Im3 = np.mean(normImage[:,:,:,6:10], axis=3)
    Im3 = Im3[:,:,:,np.newaxis]
    Im4 = np.mean(normImage[:,:,:,10:12], axis=3)
    Im4 = Im4[:,:,:,np.newaxis]
    Im5 = np.mean(normImage[:,:,:,12:15], axis=3)
    Im5 = Im5[:,:,:,np.newaxis]
    Im6 = normImage[:,:,:,15]
    Im6 = Im6[:,:,:,np.newaxis]
    
    Im = np.concatenate((Im1,Im2,Im3,Im4,Im5,Im6), axis=3)
    Im = np.swapaxes(Im, 0,2)

    image = read_image(path.join(basedir, str(number), 'DWI_reg.nii'))
    normImage = standardization(image) 
    ImDWI = np.swapaxes(normImage, 0, 2)

    return Im, ImDWI, mask, liverMask

# define network
def build_network(Inputshape1, Inputshape2, num_class):
    concat_axis = 3
    inputsDCE = layers.Input(shape = Inputshape1)
    inputsDWI = layers.Input(shape = Inputshape2)
    # DCE MRI
    #block 1
    conv1 = Conv2D(64, (3,3), activation=None, dilation_rate=1, padding='same',
                   kernel_initializer='he_uniform', name='conv1')(inputsDCE)
    conv1 = BatchNormalization(axis=-1, name='BN1')(conv1)
    conv1 = Activation(activation='relu', name='act_1')(conv1)
    
    conv1_2 = Conv2D(64, (3,3), activation=None, dilation_rate=1, padding='same',
                     kernel_initializer='he_uniform', name='conv1_2')(conv1)
    conv1_2 = BatchNormalization(axis=-1, name='BN1_2')(conv1_2)
    conv1_2 = Activation(activation='relu', name='act_1_2')(conv1_2)
    #block 2
    conv2 = Conv2D(64, (3,3), activation=None, dilation_rate=2, padding='same',
                   kernel_initializer='he_uniform', name='conv2')(conv1_2)
    conv2 = BatchNormalization(axis=-1, name='BN2')(conv2)
    conv2 = Activation(activation='relu', name='act_2')(conv2)
    
    conv2_2 = Conv2D(64, (3,3), activation=None, dilation_rate=2, padding='same',
                     kernel_initializer='he_uniform', name='conv2_2')(conv2)
    conv2_2 = BatchNormalization(axis=-1, name='BN2_2')(conv2_2)
    conv2_2 = Activation(activation='relu', name='act_2_2')(conv2_2)
    #block 3
    conv3 = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                   kernel_initializer='he_uniform', name='conv3')(conv2_2)
    conv3 = BatchNormalization(axis=-1, name='BN3')(conv3)
    conv3 = Activation(activation='relu', name='act_3')(conv3)
    
    conv3_2 = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                     kernel_initializer='he_uniform', name='conv3_2')(conv3)
    conv3_2 = BatchNormalization(axis=-1, name='BN3_2')(conv3_2)
    conv3_2 = Activation(activation='relu', name='act_3_2')(conv3_2)
    
    conv3_3 = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                     kernel_initializer='he_uniform', name='conv3_3')(conv3_2)
    conv3_3 = BatchNormalization(axis=-1, name='BN3_3')(conv3_3)
    conv3_3 = Activation(activation='relu', name='act_3_3')(conv3_3)
    #block 4
    conv4 = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                   kernel_initializer='he_uniform', name='conv4')(conv3_3)
    conv4 = BatchNormalization(axis=-1, name='BN4')(conv4)
    conv4 = Activation(activation='relu', name='act_4')(conv4)
    
    conv4_2 = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                     kernel_initializer='he_uniform', name='conv4_2')(conv4)
    conv4_2 = BatchNormalization(axis=-1, name='BN4_2')(conv4_2)
    conv4_2 = Activation(activation='relu', name='act_4_2')(conv4_2)
    
    conv4_3 = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                     kernel_initializer='he_uniform', name='conv4_3')(conv4_2)
    conv4_3 = BatchNormalization(axis=-1, name='BN4_3')(conv4_3)
    conv4_3 = Activation(activation='relu', name='act_4_3')(conv4_3)
    
    #block 5
    conv5 = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                   kernel_initializer='he_uniform', name='conv5')(conv4_3)
    conv5 = BatchNormalization(axis=-1, name='BN5')(conv5)
    conv5 = Activation(activation='relu', name='act_5')(conv5)
    
    conv5_2 = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                     kernel_initializer='he_uniform', name='conv5_2')(conv5)
    conv5_2 = BatchNormalization(axis=-1, name='BN5_2')(conv5_2)
    conv5_2 = Activation(activation='relu', name='act_5_2')(conv5_2)
    
    conv5_3 = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                     kernel_initializer='he_uniform', name='conv5_3')(conv5_2)
    conv5_3 = BatchNormalization(axis=-1, name='BN5_3')(conv5_3)
    conv5_3 = Activation(activation='relu', name='act_5_3')(conv5_3)
    
    # DWI
    #block 1
    conv1D = Conv2D(64, (3,3), activation=None, dilation_rate=1, padding='same',
                    kernel_initializer='he_uniform', name='conv1D')(inputsDWI)
    conv1D = BatchNormalization(axis=-1, name='BN1D')(conv1D)
    conv1D = Activation(activation='relu', name='act_1D')(conv1D)
    
    conv1_2D = Conv2D(64, (3,3), activation=None, dilation_rate=1, padding='same',
                      kernel_initializer='he_uniform', name='conv1_2D')(conv1D)
    conv1_2D = BatchNormalization(axis=-1, name='BN1_2D')(conv1_2D)
    conv1_2D = Activation(activation='relu', name='act_1_2D')(conv1_2D)
    #block 2
    conv2D = Conv2D(64, (3,3), activation=None, dilation_rate=2, padding='same',
                    kernel_initializer='he_uniform', name='conv2D')(conv1_2D)
    conv2D = BatchNormalization(axis=-1, name='BN2D')(conv2D)
    conv2D = Activation(activation='relu', name='act_2D')(conv2D)
    
    conv2_2D = Conv2D(64, (3,3), activation=None, dilation_rate=2, padding='same',
                      kernel_initializer='he_uniform', name='conv2_2D')(conv2D)
    conv2_2D = BatchNormalization(axis=-1, name='BN2_2D')(conv2_2D)
    conv2_2D = Activation(activation='relu', name='act_2_2D')(conv2_2D)
    #block 3
    conv3D = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                    kernel_initializer='he_uniform', name='conv3D')(conv2_2D)
    conv3D = BatchNormalization(axis=-1, name='BN3D')(conv3D)
    conv3D = Activation(activation='relu', name='act_3D')(conv3D)
    
    conv3_2D = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                      kernel_initializer='he_uniform', name='conv3_2D')(conv3D)
    conv3_2D = BatchNormalization(axis=-1, name='BN3_2D')(conv3_2D)
    conv3_2D = Activation(activation='relu', name='act_3_2D')(conv3_2D)
    
    conv3_3D = Conv2D(64, (3,3), activation=None, dilation_rate=4, padding='same',
                      kernel_initializer='he_uniform', name='conv3_3D')(conv3_2D)
    conv3_3D = BatchNormalization(axis=-1, name='BN3_3D')(conv3_3D)
    conv3_3D = Activation(activation='relu', name='act_3_3D')(conv3_3D)
    #block 4
    conv4D = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                    kernel_initializer='he_uniform', name='conv4D')(conv3_3D)
    conv4D = BatchNormalization(axis=-1, name='BN4D')(conv4D)
    conv4D = Activation(activation='relu', name='act_4D')(conv4D)
    
    conv4_2D = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                      kernel_initializer='he_uniform', name='conv4_2D')(conv4D)
    conv4_2D = BatchNormalization(axis=-1, name='BN4_2D')(conv4_2D)
    conv4_2D = Activation(activation='relu', name='act_4_2D')(conv4_2D)
    
    conv4_3D = Conv2D(64, (3,3), activation=None, dilation_rate=6, padding='same',
                      kernel_initializer='he_uniform', name='conv4_3D')(conv4_2D)
    conv4_3D = BatchNormalization(axis=-1, name='BN4_3D')(conv4_3D)
    conv4_3D = Activation(activation='relu', name='act_4_3D')(conv4_3D)
    
    #block 5
    conv5D = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                    kernel_initializer='he_uniform', name='conv5D')(conv4_3D)
    conv5D = BatchNormalization(axis=-1, name='BN5D')(conv5D)
    conv5D = Activation(activation='relu', name='act_5D')(conv5D)
    
    conv5_2D = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                      kernel_initializer='he_uniform', name='conv5_2D')(conv5D)
    conv5_2D = BatchNormalization(axis=-1, name='BN5_2D')(conv5_2D)
    conv5_2D = Activation(activation='relu', name='act_5_2D')(conv5_2D)
    
    conv5_3D = Conv2D(64, (3,3), activation=None, dilation_rate=8, padding='same',
                      kernel_initializer='he_uniform', name='conv5_3D')(conv5_2D)
    conv5_3D = BatchNormalization(axis=-1, name='BN5_3D')(conv5_3D)
    conv5_3D = Activation(activation='relu', name='act_5_3D')(conv5_3D)
    

    concat = layers.concatenate([conv1_2, conv1_2D, conv2_2, conv2_2D, conv3_2,
                                 conv3_2D, conv4_2, conv4_2D, conv5_2, conv5_2D], axis=concat_axis)
    
    #block 6
    dropout1 = Dropout(0.2)(concat)
    
    conv6 = Conv2D(128, (1,1), activation=None, dilation_rate=1,
                   kernel_initializer='he_uniform', name='conv6')(dropout1)
    conv6 = BatchNormalization(axis=-1, name='BN6')(conv6)
    conv6 = Activation(activation='relu', name='act_6')(conv6)
    
    dropout2 = Dropout(0.2)(conv6)

    
    conv6_2 = layers.Conv2D(2, (1, 1), activation='softmax')(dropout2)
    
    model = models.Model(inputs=[inputsDCE, inputsDWI], outputs=conv6_2)
    model.compile(optimizer=optimizers.Adam(lr=0.0001, decay=0.0),
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal') # lr was 0.001

    return model


def iterate_in_mb_train(train_dce, train_dwi, train_y):
    global batch_size
    
    while True:
        i_liver = np.random.choice(train_dce.shape[0], int(batch_size), replace=False)
        
        slices_input_dce = train_dce[i_liver[0],:,:,:]
        slices_input_dce = slices_input_dce[np.newaxis,:,:,:]
        
        slices_input_dwi = train_dwi[i_liver[0],:,:,:]
        slices_input_dwi = slices_input_dwi[np.newaxis,:,:,:]
        
        slices_target = train_y[i_liver[0],:,:]
        slices_target = slices_target[np.newaxis,:,:]

        lim_deg = 45 # limit of degrees of rotation
        deg = np.random.choice((range(-lim_deg,lim_deg)), int(batch_size))
        
        for p in range(1, int(batch_size)):
            p_input_dce = train_dce[i_liver[p],:,:,:]
            p_input_dwi = train_dwi[i_liver[p],:,:,:]
            p_target = train_y[i_liver[p],:,:]
            
            p_input_dce = sp.rotate(p_input_dce, deg[p], reshape=False, order=3)
            p_input_dwi = sp.rotate(p_input_dwi, deg[p], reshape=False, order=3)
            p_target = sp.rotate(p_target, deg[p], reshape=False, order=1)
                
            p_input_dce = p_input_dce[np.newaxis,:,:,:]
            p_input_dwi = p_input_dwi[np.newaxis,:,:,:]
            p_target = p_target[np.newaxis,:,:]
            
            slices_input_dce = np.concatenate((slices_input_dce, p_input_dce), axis=0)
            slices_input_dwi = np.concatenate((slices_input_dwi, p_input_dwi), axis=0)
            slices_target = np.concatenate((slices_target, p_target), axis=0)

        
            
        mb_labels = np.zeros([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 2])
        neg_target = np.ones([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 1])
        neg_target = neg_target-slices_target[:,:,:,np.newaxis]
        mb_labels[:,:,:,0:1] = neg_target
        mb_labels[:,:,:,1:2] = slices_target[:,:,:,np.newaxis]  # *5 times the weight, in this case 5.  
        
        yield [slices_input_dce, slices_input_dwi], mb_labels


def iterate_in_mb_test(test_dce, test_dwi, test_y):
    global batch_size
    
    while True:
        i_liver = np.random.choice(test_dce.shape[0], int(batch_size), replace=False)
        
        slices_input_dce = test_dce[i_liver[0],:,:,:]
        slices_input_dce = slices_input_dce[np.newaxis,:,:,:]
        
        slices_input_dwi = test_dwi[i_liver[0],:,:,:]
        slices_input_dwi = slices_input_dwi[np.newaxis,:,:,:]
        
        slices_target = test_y[i_liver[0],:,:]
        slices_target = slices_target[np.newaxis,:,:]
        
        lim_deg = 45 # limit of degrees of rotation
        deg = np.random.choice((range(-lim_deg,lim_deg)), int(batch_size))
        
        for p in range(1, int(batch_size)):
            p_input_dce = test_dce[i_liver[p],:,:,:]
            p_input_dwi = test_dwi[i_liver[p],:,:,:]
            p_target = test_y[i_liver[p],:,:]
            
            p_input_dce = sp.rotate(p_input_dce, deg[p], reshape=False, order=3)
            p_input_dwi = sp.rotate(p_input_dwi, deg[p], reshape=False, order=3)
            p_target = sp.rotate(p_target, deg[p], reshape=False, order=1)
            
                
            p_input_dce = p_input_dce[np.newaxis,:,:,:]
            p_input_dwi = p_input_dwi[np.newaxis,:,:,:]
            p_target = p_target[np.newaxis,:,:]
            
            slices_input_dce = np.concatenate((slices_input_dce, p_input_dce), axis=0)
            slices_input_dwi = np.concatenate((slices_input_dwi, p_input_dwi), axis=0)
            slices_target = np.concatenate((slices_target, p_target), axis=0)
        
            
        mb_labels = np.zeros([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 2])
        neg_target = np.ones([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 1])
        neg_target = neg_target-slices_target[:,:,:,np.newaxis]
        mb_labels[:,:,:,0:1] = neg_target
        mb_labels[:,:,:,1:2] = slices_target[:,:,:,np.newaxis] #*5  weighting of class 1 
        
        yield [slices_input_dce, slices_input_dwi], mb_labels



def bbox(img):
    img = np.sum(img, axis=0)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def imsave(fname, arr):
    sitk_img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(sitk_img, fname)
    
    
# Patient specific fine-tuning of pre-trained CNN  
def patient_specific_training(t0, t1, basedir):
    train_dce = []
    train_dwi = []
    train_y = []
    
    # Load the CNN
    LesionDetection = build_network(Inputshape1=(128, 128, 6), Inputshape2 =(128,128,3), num_class=2)
    # Load weights of pre-trained network
    LesionDetection.load_weights('H:/Docker/lesionDetectionPnetFU_DCEDWI.h5')
    
    file = path.join(basedir, str(t0),'e-THRIVE_reg.nii')
    if path.isfile(file):
        DCE, DWI, y, yLiver = load_data(basedir, nameImage='e-THRIVE_reg.nii', number=t0)
        l = np.mean(np.mean(y,axis=1), axis=1)
        lesion = np.asarray(np.where(l>0))
            
        newDCE = np.zeros((lesion.shape[1],DCE.shape[1],DCE.shape[2], DCE.shape[3]))
        newDCE = DCE[lesion[0],:,:,:]
        
        newDWI = np.zeros((lesion.shape[1],DWI.shape[1],DWI.shape[2], DWI.shape[3]))
        newDWI = DWI[lesion[0],:,:,:]
            
        newY = np.zeros((lesion.shape[1],y.shape[1],y.shape[2]))
        newY = y[lesion[0],:,:]
              
        DCE4 = newDCE[:, 50:178, 40:168, :]
        DCE4 = np.concatenate((DCE4, newDCE[:, 30:158, 20:148, :]),axis=0)
        DCE4 = np.concatenate((DCE4, newDCE[:, 70:198, 20:148, :]),axis=0)
        DCE4 = np.concatenate((DCE4, newDCE[:, 30:158, 60:188, :]),axis=0)
        DCE4 = np.concatenate((DCE4, newDCE[:, 70:198, 60:188, :]),axis=0)
        
        DWI4 = newDWI[:, 50:178, 40:168, :]
        DWI4 = np.concatenate((DWI4, newDWI[:, 30:158, 20:148, :]),axis=0)
        DWI4 = np.concatenate((DWI4, newDWI[:, 70:198, 20:148, :]),axis=0)
        DWI4 = np.concatenate((DWI4, newDWI[:, 30:158, 60:188, :]),axis=0)
        DWI4 = np.concatenate((DWI4, newDWI[:, 70:198, 60:188, :]),axis=0)
            
        y4 = newY[:, 50:178, 40:168]
        y4 = np.concatenate((y4, newY[:, 30:158, 20:148]),axis=0)
        y4 = np.concatenate((y4, newY[:, 70:198, 20:148]),axis=0)
        y4 = np.concatenate((y4, newY[:, 30:158, 60:188]),axis=0)
        y4 = np.concatenate((y4, newY[:, 70:198, 60:188]),axis=0)
            
        if not np.any(train_dce):
            train_dce = DCE4
            train_dwi = DWI4
            train_y = y4
#            train_dce = newDCE
#            train_dwi = newDWI
#            train_y = newY
        else:
            train_dce = np.concatenate((train_dce, DCE4), axis=0)
            train_dwi = np.concatenate((train_dwi, DWI4), axis=0)
            train_y = np.concatenate((train_y, y4), axis=0)
    
    train_dce = np.asarray(train_dce)
    train_dwi = np.asarray(train_dwi)
    train_y = np.asarray(train_y)
    print('Loading training data')
    

    batch_size = 4 #4
    
    i=0
    for layer in LesionDetection.layers:
        layer.trainable = False
        i = i+1
#        print(i,layer.name)
        if i == 76:
            break
    
    n_epochs = 100 
    LesionDetection.fit_generator(iterate_in_mb_train(train_dce, train_dwi, train_y), 1,
                                  epochs=n_epochs, verbose=0)
    
    #                                callbacks=[tbCallback],
    LesionDetection.save_weights('./lesionDetectionPnet_patientSpecific.h5', overwrite=True)
    

    
    
BaselineScans = [7, 9, 12, 16, 20, 22, 24, 28, 41, 44, 52, 54, 56, 59, 61, 63]
FollowUpScans = [8, 10, 13, 17, 21, 23, 25, 29, 42, 45, 53, 55, 57, 60, 62, 64]

for sub in range(0, len(BaselineScans)):
    
    batch_size = 4
    # Load the data with selected slices
    basedir = r'C:\Users\user\Documents\FineTuningData\Testing'
    t0 = BaselineScans[sub] # baseline scans
    t1 = FollowUpScans[sub] # follow-up scans

    # Patient-specific fine-tuning
    patient_specific_training(t0, t1, basedir)
    
    LesionDetection = build_network(Inputshape1=(256, 256, 6), Inputshape2 =(256,256,3), num_class=2)
    LesionDetection.load_weights('./lesionDetectionPnet_patientSpecific.h5')
    
    # Get the probalities of the baseline scan
    file = path.join(basedir, str(t1),'e-THRIVE_reg.nii')
    if path.isfile(file):
        inputDCE, inputDWI, _ , _ = load_data(basedir, nameImage='e-THRIVE_reg.nii', number=t1)

        test_DCE = inputDCE[0,:,:,:]
        test_DCE = test_DCE[np.newaxis, :, :, :]
        test_DWI = inputDWI[0,:,:,:]
        test_DWI = test_DWI[np.newaxis, :, :, :]
        predict_50 = LesionDetection.predict([test_DCE, test_DWI])
        predict_50 = predict_50[0,:,:,:]
        prediction = predict_50[:,:,1]*100
        prediction = prediction[np.newaxis,:,:]
           
        for slice in range(1,inputDCE.shape[0]):
            test_DCE = inputDCE[slice,:,:,:]
            test_DCE = test_DCE[np.newaxis, :, :, :]
            test_DWI = inputDWI[slice,:,:,:]
            test_DWI = test_DWI[np.newaxis, :, :, :]
            predict_50 = LesionDetection.predict([test_DCE, test_DWI])
            predict_50 = predict_50[0,:,:,:]
            predict_50 = predict_50[:,:,1]*100
            prediction = np.concatenate((prediction, predict_50[np.newaxis,:,:]), axis=0)
               
        prediction = np.asarray(prediction, dtype='int')
#        imsave('C:/Users/user/Documents/Detection/FollowUpDetection results/Testing results/result_'+str(t1)+'_Pnet_DCEDWI_FT.nii', prediction)
        imsave('C:/Users/user/Documents/Detection/FollowUpDetection results/Testing results/FTiter50/result_'+str(t1)+'.nii', prediction)


    

