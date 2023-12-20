# This script allows us to select the amplitudes using score-to-bin, cluster-bins, large amplitude and electron correlation schemes

__authors__ = 'Varuna Pathirage and Justin Phillips'


# Importing packages
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# This function calculates the feature weights
def calc_final_factor(X_train, y_train):
    # X_train: feature array
    # y_train: CCSD t2 amplitude array
    Bigmat = np.concatenate([X_train, y_train], axis=1)
    corr_mat = np.corrcoef(Bigmat.T)
    finalfactor = np.nan_to_num(corr_mat[30][:30], nan=0.0)
    
    return finalfactor


# This function selects amplitudes using score-to-bins approach for training
def SB_amp_selection(X_train, y_train, p_amps=0.2, n_bins=100):
    # X_train: feature array
    # y_train: CCSD t2 amplitude array
    # p_amps: percent of amplitudes that needed to be selected (as a decimal): default=0.2 (20%)
    # n_bins: number of bins: default=100 
    
    finalfactor = calc_final_factor(X_train, y_train)
    scaled_features = MinMaxScaler().fit_transform(X_train)
    sq_features = np.square(scaled_features)
    tile_weights = np.tile(finalfactor, (y_train.size,1))
    sq_weights = np.square(tile_weights)
    feature_scores = np.sqrt(np.sum((np.multiply(sq_features, sq_weights)), axis = 1))
    
    Dist, x_lim, y_lim = np.histogram2d(np.ravel(y_train), feature_scores, bins=n_bins)
    
    #Expanding limits to include all amps
    x_lim[0] = x_lim[0]-(abs(x_lim[0]-x_lim[1]))/100
    y_lim[0] = y_lim[0]-(abs(y_lim[0]-y_lim[1]))/100
    x_lim[-1] = x_lim[-1]+(abs(x_lim[-1]-x_lim[-2]))/100
    y_lim[-1] = y_lim[-1]+(abs(y_lim[-1]-y_lim[-2]))/100
    
    big_cut = (np.round(X_train.shape[0]*p_amps))
    bin_cut = 0
    tot_amps = 0
    while tot_amps < big_cut:
        tot_amps = np.sum(np.where(Dist<bin_cut, Dist, bin_cut))
        bin_cut += 1
    bin_cut = (bin_cut-2)
    
    positions = np.arange(y_train.shape[0])
    np.random.shuffle(positions)
    data_frequency = np.zeros((x_lim.size-1, y_lim.size-1))
    Bigamps = []
    Bigfeatures = []
    for pos in positions:
        amplitude = y_train[pos]
        feature = X_train[pos]
        feature_score = feature_scores[pos]
        for x in range(0, x_lim.size-1):
            if amplitude >= x_lim[x] and amplitude < x_lim[x+1]:
                for y in range(0, y_lim.size-1):
                    if feature_score >= y_lim[y] and feature_score < y_lim[y+1]:
                        data_frequency[x,y] += 1
                        if data_frequency[x,y] <= bin_cut:
                            Bigfeatures.append(feature)
                            Bigamps.append(amplitude)
    
    return np.asarray(Bigfeatures), np.asarray(Bigamps)



# This function selects amplitudes using cluster-bins approach for training
def CB_amp_selection(X_train, y_train, p_amps=0.2, n_bins=100):
    # X_train: feature array
    # y_train: CCSD t2 amplitude array
    # p_amps: percent of amplitudes that needed to be selected (as a decimal): default = 0.2 (20%)
    # n_bins: number of bins: default = 100
    
    finalfactor = calc_final_factor(X_train, y_train)
    
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    for a in range(0,30):
        X_train_scaled[:,a] *= finalfactor[a]

  
    Bigmat = np.concatenate([X_train_scaled, y_train], axis=1)
    kmeans = KMeans(n_clusters=n_bins, random_state=0).fit(Bigmat)
    labels = kmeans.labels_
    
    big_cut = (int(np.round(y_train.shape[0]*p_amps)))
        
    Dist, x_lims = np.histogram(labels, bins=n_bins)
    
    bin_cut = 0
    tot_amps = 0
    while tot_amps < big_cut:
        tot_amps = np.sum(np.where(Dist<bin_cut, Dist, bin_cut))
        bin_cut += 1
    bin_cut = (bin_cut-2)
    
    x_lims = np.arange(n_bins)
    
    position = np.arange(y_train.shape[0])
    np.random.shuffle(position)
    Bigfeatures = []
    Bigamps = []
    data_frequency = np.zeros(x_lims.size)
    for pos in position:
        amplitude = y_train[pos]
        feature = X_train[pos]
        label_amp = labels[pos]
        for x in x_lims:
            if label_amp == x:
                data_frequency[x] += 1 
                if data_frequency[x] <= bin_cut:
                    Bigfeatures.append(feature)
                    Bigamps.append(amplitude)
                    
    return np.asarray(Bigfeatures), np.asarray(Bigamps)



# This function selects amplitudes using large amplitude approach for training
def LA_amp_selection(X_train, y_train, cutoff=0.0001):
    # X_train: feature array
    # y_train: CCSD t2 amplitude array
    # cutoff: cutoff for mp2 amplitudes: default = 0.0001
    
    mp2_amps = np.abs(X_train[:,23])
    
    return X_train[mp2_amps>=cutoff], y_train[mp2_amps>=cutoff]

# This function selects amplitudes using large amplitude approach for testing
def LA_amp_selection_testing(X_test, cutoff=0.0001):
    # X_train: feature array
    # cutoff: cutoff for mp2 amplitudes
    
    test_features = []
    test_positions = []
    for i in range(0,X_test.shape[0]):
        feature = X_test[i]
        if np.abs(feature[23]) >= cutoff:
            test_positions.append(i)
            test_features.append(feature)
    
    Testfeatures = np.asarray(test_features)
    Testpositions = np.asarray(test_positions)
    
    return Testfeatures, Testpositions



# This function selects amplitudes using electron correlation approach for training
def EC_amp_selection(X_train, y_train, cutoff=1e-7):
    # X_train: feature array
    # y_train: CCSD t2 amplitude array
    # cutoff: cutoff for electron correlation: default = 1e-7
    
    corr_e = np.abs(X_train[:,23]*X_train[:,22])
    
    return X_train[corr_e>=cutoff], y_train[corr_e>=cutoff]

# This function calculates cutoffs for electron correlation approach for testing
def calc_EC_cutoff(X_train, p_cutoff = 99.15):
    # X_train: feature array
    # p_cutoff: cutoff for electron correlation: default = 99.15 (99.15%)
    
    corr_e = X_train[:,22]*X_train[:,23]
    corr_e_list = np.ndarray.tolist(corr_e)
    corr_e_list.sort()
    total_os_corr_e = np.sum(corr_e)
    os_corr_e = 0.0
    total_amps = len(corr_e_list)
    for a in range(len(corr_e_list)):
        os_corr_e += corr_e_list[a]
        percent = os_corr_e/total_os_corr_e*100
        if percent > p_cutoff:
            corr_lim = corr_e_list[a]
            break
    
    return corr_lim

# This function selects amplitudes using electron correlation approach for testing
def EC_amp_selection_testing(X_test, cutoff):
    # X_train: feature array
    # cutoff: cutoff for electron correlation
    
    test_features = []
    test_positions = []
    for i in range(0,X_test.shape[0]):
        feature = X_test[i]
        if (feature[23]*feature[22]) <= cutoff:
            test_positions.append(i)
            test_features.append(feature)
    
    Testfeatures = np.asarray(test_features)
    Testpositions = np.asarray(test_positions)
    
    return Testfeatures, Testpositions



# This function selects amplitudes using probabilistic selection approach for training
def PS_amp_selection(X_train, y_train, p_amps = 0.2):
    # X_train: feature array
    # y_train: CCSD t2 amplitude array
    # p_amps: percent of amplitudes that needed to be selected (as a decimal): default = 0.2 (20%)
    
    AllAmps = y_train
    AllFeats = X_train
    AllComb = np.concatenate((AllAmps, AllFeats), axis = 1)
    Feat1 = np.hsplit(AllFeats, 30)[8]
    Feat2 = np.hsplit(AllFeats, 30)[12]
    MP2 = np.hsplit(AllFeats, 30)[23]
    MultProb = []
    # Calculate the probability
    for Indice in range(len(AllAmps)):
        MultProb.append(abs(MP2[Indice]*(Feat1[Indice]+Feat2[Indice])))
    #Reshape the probility array
    Prob = np.squeeze(np.asarray(MultProb))
    #Normalize probabilities
    AdjFactor = np.sum(Prob)
    AdjProb = Prob/AdjFactor
    
    #Determining the number of amps
    NumChosenAmps = np.round(np.size(Prob)*p_amps,decimals=0).astype(int)
    
    #Creating indicies for each amp
    IndexAllAmps = np.arange(np.size(Prob))
    
    #Randomly select the amps based on the probability
    IndexAdjAmps = np.random.choice(IndexAllAmps,size=NumChosenAmps,replace=False,p=AdjProb)
    
    #Separating the chosen amp indicies
    SubIndexAdjComb = np.delete(IndexAllAmps,IndexAdjAmps)
    ReducedComb = np.delete(AllComb,SubIndexAdjComb,axis=0)
    
    
    #Spliting back the amps and features
    ReducedAmps = np.hsplit(ReducedComb, [1,31])[0]
    ReducedFeats = np.hsplit(ReducedComb, [1,31])[1]
    
    return ReducedFeats, ReducedAmps
