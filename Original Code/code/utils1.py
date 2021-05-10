import numpy as np
def tosamples0(ecg):
    if len(ecg[0])<625:
        new = np.concatenate((ecg,np.zeros((12,625-len(ecg[0])))))
    else:
        i = 0
        new = []
        while i+625<len(ecg[0]):
            new.append(ecg[:,i:i+625])
            i += 312
        new.append(ecg[:,-625:])
    samples = np.array(new)
    return samples
def tosamples(ecg):
    if len(ecg[0])<625:
        new = np.concatenate((ecg,np.zeros((1,625-len(ecg[0])))))
    else:
        i = 0
        new = []
        while i+625<len(ecg[0]):
            new.append(ecg[:,i:i+625])
            i += 156
        new.append(ecg[:,-625:])
    samples = np.array(new)
    return samples.reshape(-1,1,625,1)
def remove_outlier(ecg):
    if (ecg>3).any():
        for i in range(12):
            b = np.argwhere(np.diff(ecg[i])>3)
            if b.shape[0]>0:
                for k in b[:,0]:
                    ecg[i][k+1] = ecg[i][k]
    if (ecg<-3).any():
        for i in range(12):
            b = np.argwhere(np.diff(ecg[i])<-3)
            if b.shape[0]>0:
                for k in b[:,0]:
                    ecg[i][k+1] = ecg[i][k]
    return ecg
def remove_noise(ecg):
    le = ecg.shape[1]
    for j in range(12):
        noise = []
        b = np.argwhere(np.abs(ecg[j])>2.5)
        b = b[:,0]
        c = np.diff(b)
        count = 0
        pn = 0
        for k in range(len(c)):
            if c[k]<2:
                count += 1
                if count>=8:
                    noise.append(b[k+1])
            elif c[k]>1 and c[k]<8:
                count = 0
                pn += 1
                if pn>1:
                    noise.append(b[k+1])
            elif c[k]<25:
                count = 0
                pn = 0
                noise.append(b[k+1])
            else:
                count = 0
                pn = 0
        if len(noise)>0:
            pre = -1
            for l in range(len(noise)):
                if pre>=0 and noise[l]-pre<200:
                    be = noise[l-1]
                else:
                    be = max(0,noise[l]-60)
                en = min(le,noise[l]+60)
                pre = noise[l]
                ecg[j][be:en] = 0
    return ecg
def getsamples0(ecg):
    length = ecg.shape[1]
    ecg = ecg[:,:int(length//4*4)]
    ecg = ecg.reshape(ecg.shape[0],-1,4)
    ecg = np.average(ecg,axis=2)
    ecg = remove_outlier(ecg)
    ecg = remove_noise(ecg)
    samples = tosamples0(ecg)
    return samples.reshape(-1,12,625,1)
def getsamples(ecg):
    length = ecg.shape[1]
    ecg = ecg[:,:int(length//4*4)]
    ecg = ecg.reshape(ecg.shape[0],-1,4)
    ecg = np.average(ecg,axis=2)
    ecg = remove_outlier(ecg)
    ecg = remove_noise(ecg)
#    samples = tosamples(ecg)
#    return samples.reshape(-1,12,625,1)
    return ecg

def getsample1s(ecg):
    length = ecg.shape[1]
    ecg = ecg[:,:int(length//4*4)]
    ecg = ecg.reshape(ecg.shape[0],-1,4)
    ecg = np.average(ecg,axis=2)
    ecg = remove_outlier(ecg)
    ecg = remove_noise(ecg)
    return ecg[:,:1000]