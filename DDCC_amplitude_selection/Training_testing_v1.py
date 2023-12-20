#!/usr/bin/env python
# coding: utf-8

# In[1]:


import psi4
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from helper_CC_ML_pert_T_fc_lo import *
import os
from Amplitude_selection import *
MLt2=0


# In[2]:


#Set this to the number of cores on your computer, if you run it on the head node, use 1
psi4.set_num_threads(1)


# In[3]:


#This box contains all the features the CCSD module logs for us to make predictions
#Ultimately, just make sure you run this every time and all is well.

features = ['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1',
            'Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2', 'diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag', 'Kia1mag', 'Kia2mag']
'''
Key:
Letters:
E-Energy of the orbital
H-1e contribution to the orbital energy
J-Coulombic contribution to orbital energy
K-Exchange contribution to orbital energy
Placement:
occ or virt, you get this..
Number:
is it electron one or two from the two electron excitation


Jia1- coulomb integral between orbital occ1 and vir1
Jia2 " but 2
Kia1 - exchange integral between orbital 
Kia2 Same but exchange integral
diag - is it on the diagonal, aka, are the two excited electrons going to the same orbital **this is important fyi
orbdiff - (Evir2 + Evir1 - Eocc1 - Eocc2)
doublecheck - full 2electron integral
t2start - INITIAL MP2 amplitude **this is the inital guess
t2mag - np.log10(np.absolute(t2start)) ~ this is going to be a common trend, since it is more straightforward for ML algorithms to understand
t2sign - (t2start > 1)? 
Jia1mag - np.log10(np.absolute(feature))
Jia2mag np.log10(np.absolute(feature))
Kia1mag  np.log10(np.absolute(feature))
Kia2mag np.log10(np.absolute(feature))

'''


# In[4]:


#This functional gets all the amplitudes from some reference set
def GetAmps(Foldername, occ=False, vir=False, criterion='none'):
    i=1
    for filename in os.listdir(str(Foldername)):
        if filename.endswith('.xyz'):
            psi4.core.clean()
            path1=str(str(Foldername)+filename)
            text = open(path1, 'r').read()
            #print(text)
            mol = psi4.geometry(text)
            psi4.core.clean()

            psi4.set_options({'basis':        'STO-3G',#'cc-pVDZ', 'aug-cc-pVDZ',
                              'scf_type':     'pk',
                              'reference':    'rhf',
                              'mp2_type':     'conv',
                              'e_convergence': 1e-8,
                              'Freeze_core':   False,
                              'd_convergence': 1e-8})

            A=HelperCCEnergy(mol)

            A.compute_energy()
            matrixsize=(A.nocc-A.nfzc)*(A.nocc-A.nfzc)*A.nvirt*A.nvirt
            Bigmatrix=np.zeros([matrixsize, len(features)])
            for x in range(0,len(features)):
                Bigmatrix[:, x]=getattr(A, features[x]).reshape(matrixsize)
            Bigamp=A.t2.reshape(matrixsize,1)
            
            if criterion.lower() == 'ps':
                if i==1:
                    Redamps, Redfeatures = PS_amp_selection(Bigmatrix, Bigamp)
                    Bigamps=Redamps
                    Bigfeatures=Redfeatures
                    i=2
                else:
                    Redamps, Redfeatures = PS_amp_selection(Bigmatrix, Bigamp)
                    Bigfeatures=np.vstack((Bigfeatures,Redfeatures))
                    Bigamps=np.vstack((Bigamps,Redamps))
                    
            elif criterion.lower() == 'la':
                if i==1:
                    Bigfeatures, Bigamps = LA_amp_selection(Bigmatrix, Bigamp)
                    i=2
                else:
                    Redfeatures, Redamps = LA_amp_selection(Bigmatrix, Bigamp)
                    Bigfeatures=np.vstack((Bigfeatures,Redfeatures))
                    Bigamps=np.vstack((Bigamps,Redamps))
                    
            elif criterion.lower() == 'ec':
                if i==1:
                    Bigfeatures, Bigamps = EC_amp_selection(Bigmatrix, Bigamp)
                    i=2
                else:
                    Redfeatures, Redamps = EC_amp_selection(Bigmatrix, Bigamp)
                    Bigfeatures=np.vstack((Bigfeatures,Redfeatures))
                    Bigamps=np.vstack((Bigamps,Redamps))
            
            else:
                if i==1:
                    X_train=Bigmatrix
                    y_train=Bigamp
                    i=2
                else:
                    X_train=np.vstack((X_train,Bigmatrix))
                    y_train=np.vstack((y_train,Bigamp))
    
    if criterion.lower() == 'sb':
        Bigfeatures, Bigamps = SB_amp_selection(X_train, y_train)
    
    elif criterion.lower() == 'cb':
        Bigfeatures, Bigamps = CB_amp_selection(X_train, y_train)
    
    elif criterion.lower() != 'ps' and criterion.lower() != 'la' and criterion.lower() != 'ec':
        Bigfeatures = X_train
        Bigamps = y_train
    
    return Bigfeatures, Bigamps



#This function tests the model

def Test(Foldername, occ=False, vir=False, criterion = 'none'):
    steps=list()
    difference=list()
    supalist=list()
    startenergy=list()
    finalenergy=list()
    filenames=list()
    rhfenergy=list()
    pert_t_energy_start = list()
    pert_t_energy_final = list()
    total_e_start_array = list()
    total_e_final_array = list()

    for filename in os.listdir(Foldername):
        if filename.endswith('.xyz'): 
            psi4.core.clean()
            filenames.append(filename)
            print ("filename is "+filename)
            path1=str(Foldername+filename)
            text = open(path1, 'r').read()
            #print(text)
            mol = psi4.geometry(text)

            psi4.set_options({'basis':        'STO-3G', #'cc-pVDZ', 'aug-cc-pVDZ',
                              'scf_type':     'pk',
                              'maxiter':      1000,
                              'reference':    'rhf',
                              'mp2_type':     'conv',
                              'e_convergence': 1e-8,
                              'Freeze_core':   False,
                              'd_convergence': 1e-8})
                
            MLt2=0
            A=HelperCCEnergy(mol)
                
            matrixsize=(A.nocc-A.nfzc)*(A.nocc-A.nfzc)*A.nvirt*A.nvirt
            Xnew=np.zeros([1,matrixsize,len(features)])
            for x in range (0,len(features)):
                Xnew[0,:,x]=getattr(A, features[x]).reshape(matrixsize)

            Xnew=np.reshape(Xnew, (matrixsize,len(features)))
                
            if criterion.lower() == 'la':
                Testfeats, Testpos = LA_amp_selection_testing(Xnew)
                
                X_new_scaled= scaler.transform(Testfeats)
                X_newer_scaled= X_new_scaled
            
                for x in range (0,len(features)):
                    X_newer_scaled[:,x] *= finalfactor[x]
            
                ynew2=ml.predict(X_newer_scaled)
                newamps = np.zeros((Xnew.shape[0],1))
                for i in range(0,Testfeats.shape[0]):
                    amp = ynew2[i]
                    pos = Testpos[i]
                    newamps[pos] = amp

            elif criterion.lower() == 'ec':
                cut = calc_EC_cutoff(Xnew)
                Testfeats, Testpos = EC_amp_selection_testing(Xnew, cutoff=cut)
                X_new_scaled= scaler.transform(Testfeats)
                X_newer_scaled= X_new_scaled
            
                for x in range (0,len(features)):
                    X_newer_scaled[:,x] *= finalfactor[x]
            
                ynew2=ml.predict(X_newer_scaled)
                newamps = np.zeros((Xnew.shape[0],1))
                for i in range(0,Testfeats.shape[0]):
                    amp = ynew2[i]
                    pos = Testpos[i]
                    newamps[pos] = amp    
            
            else:
                X_new_scaled= scaler.transform(Xnew)
                X_newer_scaled= X_new_scaled
                    
                for x in range (0,len(features)):
                    X_newer_scaled[:,x] *= finalfactor[x]
    
                newamps=ml.predict(X_newer_scaled)
        
            MLt2=newamps.reshape(A.nocc-A.nfzc,A.nocc-A.nfzc,A.nvirt,A.nvirt)
            A.t2=MLt2
                
            A.compute_t1()
            pert_t_e_start = A.compute_pert_t()
            A.compute_energy()
            rhfenergy.append(A.rhf_e)
            startenergy.append(A.StartEnergy)
            finalenergy.append(A.FinalEnergy)
            pert_t_e_final = A.compute_pert_t()
                
            total_e_start = A.rhf_e + A.StartEnergy + pert_t_e_start
            total_e_final = A.rhf_e + A.FinalEnergy + pert_t_e_final
                
            pert_t_energy_start.append(pert_t_e_start)
            pert_t_energy_final.append(pert_t_e_final)
                
            total_e_start_array.append(total_e_start)
            total_e_final_array.append(total_e_final)
                
    
    difference.append(sum( np.abs(np.asarray(startenergy) - np.asarray(finalenergy))) /len(startenergy))
    
    error_mat_pert_t = np.absolute(np.asarray(total_e_start_array) - np.asarray(total_e_final_array))
    MAE_pert_t = np.average(error_mat_pert_t)
    
    print ('Filenames')
    print (filenames)
    print ('Start Energy')
    print (np.add(np.array(startenergy),np.array(rhfenergy)))
    print ('Individual CCSD Differences')
    print (np.abs(np.asarray(startenergy) - np.asarray(finalenergy)))
    
    print ('Average CCSD Differences')
    print (difference)
    
    print ('Individual CCSD(T) Differences')
    print (np.abs(np.asarray(total_e_start_array) - np.asarray(total_e_final_array)))
    
    print ('Average CCSD(T) Differences')
    print (MAE_pert_t)


# In[5]:


X_train,y_train=GetAmps('Training/', criterion='La')


# In[6]:


# Double checking the number of amps

y_train.shape[0]/np.load('All_LO.npz')['amps'].shape[0]*100


# In[7]:


#For adaptive weights

finalfactor = calc_final_factor(X_train, y_train)


# In[8]:


# Scale all data before using them as features

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

#This multiplies by the weighting vector from the start
for a in range(0,len(features)):
    X_train_scaled[:,a] *= finalfactor[a]

#This trains the model with our data    
ml=(RandomForestRegressor(n_estimators=350, n_jobs=-1).fit(X_train_scaled,y_train.reshape(-1,)))


# In[9]:


Test('Testing/', criterion='LA')


# In[ ]:




