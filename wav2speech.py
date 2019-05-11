import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import PCA
from python_speech_features import logfbank
import pickle
import keras
import matplotlib.pyplot as plt
import json


# /Users/Shared/anaconda/envs/wav2speech

# fname = 'wav/obama.wav'
# fname = 'wav/portman.wav'
# fname = 'wav/laughing.wav'

name = 'sigg_sample'

fname = 'wav/{}.wav'.format(name)
model_file = 'model/jm_split_a_08_model.h5'
model_weights = 'model/jm_split_a_08_weights_epoch_020.h5'
pca_model = 'model/pca_deforms_a.pkl'
json_in = 'model/scene_a_01_0184.json'

with open(json_in, 'r') as fid:
    data = json.load(fid)
tris = data['triangulation']

xmean = np.array([-10.37841939, -10.29596543,  -9.52635941,  -8.62898119,
                  -8.69664673,  -8.21468825,  -7.82792466,  -7.63330412,
                  -7.16496604,  -7.01107227,  -7.13848479,  -7.27367869,
                  -7.39360228,  -7.80416838,  -8.12128764,  -8.38547911,
                  -8.61007819,  -8.45082173,  -8.26495103,  -8.2375674,
                  -8.28130973,  -8.35239667,  -8.49759591,  -8.69153259,
                  -8.74377098,  -8.71080804,  -8.70976421,  -8.89688068,
                  -9.20472392,  -9.14222207,  -8.77120357,  -8.60124022,
                  -8.64800577,  -8.71983164,  -9.03468397,  -9.50371695,
                  -9.7947639,  -9.87434974,  -9.98449006, -10.2410216])


xscale = np.array([1.58813686, 1.83807591, 2.13763173, 2.6464941, 2.82898367,
                   2.95084312, 3.0551267, 3.2245095, 3.31486539, 3.40277908,
                   3.539413, 3.6050623, 3.62343044, 3.59356218, 3.54632535,
                   3.5146207, 3.46488724, 3.42945575, 3.40136951, 3.3930116,
                   3.33723809, 3.28875783, 3.2595026, 3.19884305, 3.18120324,
                   3.21133435, 3.19104206, 3.05393208, 2.89993749, 2.88674008,
                   2.97305169, 3.04128255, 3.09865122, 3.15371995, 3.17500542,
                   3.23045878, 3.30656913, 3.31711384, 3.27755814, 3.23892092])


ymean = np.array([-6.42464600e-05,  2.06111677e-05,  1.18896142e-04,
                -5.87192199e-04, 1.39384340e-05, -9.03689058e-04,
                -3.23345362e-04, 1.35912019e-03, -3.69690836e-03,
                -2.11017749e-03,  7.85311141e-03, -1.00601867e-08,
                -9.06127048e-11, -2.00905313e-08])


yscale = np.array([0.99999785, 0.9999983, 0.99999731, 0.99999767, 0.99999848,
                   0.99999795, 0.99999717, 0.99999662, 5.92231376, 2.68581913,
                   3.66035934, 2.53686941, 1.72943674, 5.64413309])


with open(pca_model, 'rb',) as fid:
    pca = pickle.load(fid, encoding='latin1')


def split_modes(Y):
    _pca = Y[0].squeeze() * yscale[:8] + ymean[:8]
    trxyz = Y[1].squeeze() * yscale[8:] + ymean[8:]
    deforms = pca.inverse_transform(_pca).reshape(-1, 120, 3)

    print(deforms.shape)
    rotations = trxyz[:, :3]
    rotations -= rotations.mean(0)
    translations = trxyz[:, 3:]
    return deforms, rotations, translations


kw = dict(
    samplerate=16000,
    winlen=2 / 59.94,
    winstep=1 / 59.94,
    nfilt=40,
    nfft=1024,
    lowfreq=0,
    highfreq=None,
    preemph=0.97)


def wav2predict(wav_fname):
    model = keras.models.load_model(model_file)
    model.load_weights(model_weights)
    fs, sig = wavfile.read(fname)
    sig = sig.astype(np.float32) / sig.max()
    if fs != 16000:
        print('Wav file must be 16kHz')
        return
    print(fs, sig.shape)
    X = logfbank(sig, **kw).reshape(1, -1, 40)
    X = (X - xmean) / xscale
    Y = model.predict(X)
    deforms, rotations, translations = split_modes(Y)

    return deforms, rotations, translations


def plot(rotations, translations):
    labs = ['x', 'y', 'z']
    plt.figure()
    for i in range(3):
        plt.plot(rotations[:, i], label=labs[i])
        plt.title('rotation')
        plt.legend()
    plt.figure()
    for i in range(3):
        plt.plot(translations[:, i], label=labs[i])
        plt.title('translation')
        plt.legend()
    plt.show()


deforms, rotation, translation = wav2predict(fname)

d = dict(deforms=deforms.tolist(),
         rotation=rotation.tolist(),
         translation=translation.tolist(),
         triangulation=tris)

with open('{}.json'.format(name), 'w') as fid:
    json.dump(d, fid)
