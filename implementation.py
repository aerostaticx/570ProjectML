import torch.nn as nn
import torch.nn.functional as F
import os
import csv
import string 
import re
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.optim as optim
from PIL import Image
import webrtcvad
from mfcc import *
import torchvision.utils as vutils
from vad import read_wave, write_wave, frame_generator, vad_collector
import shutil
from scipy.io import wavfile

#Network definitions
class voiceEmbeddingNetwork(nn.Module):
    def __init__(self):
        super(voiceEmbeddingNetwork, self).__init__()
        self.conv1 = nn.Conv1d(64,256,3,2,1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256,384,3,2,1, bias=False)
        self.bn2 = nn.BatchNorm1d(384)
        self.conv3 = nn.Conv1d(384,576,3,2,1, bias=False)
        self.bn3 = nn.BatchNorm1d(576)
        self.conv4 = nn.Conv1d(576,864,3,2,1, bias=False)
        self.bn4 = nn.BatchNorm1d(864)
        self.conv5 = nn.Conv1d(864,64,3,2,1, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = self.conv5(x)
        x = F.avg_pool1d(x, x.size()[2], stride = 1)
        x = x.view(x.size()[0], -1, 1, 1)
        return x

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(64,1024,4,1,0)
        self.dconv2 = nn.ConvTranspose2d(1024,512,4,2,1)
        self.dconv3 = nn.ConvTranspose2d(512,256,4,2,1)
        self.dconv4 = nn.ConvTranspose2d(256,128,4,2,1)
        self.dconv5 = nn.ConvTranspose2d(128,64,4,2,1)
        self.dconv6 = nn.ConvTranspose2d(64,3,1,1,0)

    def forward(self, x):
        x = self.dconv1(x)
        x = nn.functional.relu(x)
        x = self.dconv2(x)
        x = nn.functional.relu(x)
        x = self.dconv3(x)
        x = nn.functional.relu(x)
        x = self.dconv4(x)
        x = nn.functional.relu(x)
        x = self.dconv5(x)
        x = nn.functional.relu(x)
        x = self.dconv6(x)
        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3,32,1,1,0)
        self.conv2 = nn.Conv2d(32,64,4,2,1)
        self.conv3 = nn.Conv2d(64,128,4,2,1)
        self.conv4 = nn.Conv2d(128,256,4,2,1)
        self.conv5 = nn.Conv2d(256,512,4,2,1)
        self.conv6 = nn.Conv2d(512,64,4,1,0)
        self.fc = nn.Linear(64, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv4(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv5(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv6(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return torch.sigmoid(x)

class classifier(nn.Module):
    def __init__(self, classifierOutputChannel):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(3,32,1,1,0)
        self.conv2 = nn.Conv2d(32,64,4,2,1)
        self.conv3 = nn.Conv2d(64,128,4,2,1)
        self.conv4 = nn.Conv2d(128,256,4,2,1)
        self.conv5 = nn.Conv2d(256,512,4,2,1)
        self.conv6 = nn.Conv2d(512,64,4,1,0)
        self.fc = nn.Linear(64, classifierOutputChannel, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv4(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv5(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv6(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, 1)

class voiceCustomDataset(Dataset):
    def __init__(self, voiceList):
        self.voiceList = voiceList
    
    def __getitem__(self, ind): # required to override so dataset can get i-th sample of voicedata/label
        voiceNPY = np.load(self.voiceList[ind][0]) # load npy file
        ################################################################################# author code
        voiceNPY = voiceNPY.T.astype('float32')
        voiceLabel = self.voiceList[ind][2]
        pt = np.random.randint(voiceNPY.shape[1] - 800 + 1)
        voiceNPY = voiceNPY[:, pt:pt+800]
        #################################################################################
        return voiceNPY, voiceLabel

    def __len__(self): #required to override to return size of dataset
        return len(self.voiceList)


class faceCustomDataset(Dataset):
    def __init__(self, faceList):
        self.faceList = faceList
    
    def __getitem__(self, ind):
        ################################################################################# author code
        faceImage = Image.open(self.faceList[ind][0]) 
        faceImage = faceImage.convert('RGB').resize([64, 64])
        faceImage = np.transpose(np.array(faceImage), (2, 0, 1))
        faceImage = ((faceImage - 127.5) / 127.5).astype('float32') # normalize
        faceLabel = self.faceList[ind][2]
        if np.random.random() > 0.5:
           faceImage = np.flip(faceImage, axis=2).copy()
        #################################################################################
        return faceImage, faceLabel

    def __len__(self):
        return len(self.faceList) 


def cycle(iterable):
    while True:
        for (x, xlabel) in iterable:
            yield x, xlabel

#Loading data

# First need to get celeb names and corresponding celeb IDs from csv
DataPath = os.path.expanduser('/home/oem/570Project') #1 MODIFY
def getCelebID():
    fp = os.path.join(DataPath, 'vox1_meta.csv')
    with open(fp, 'r') as f:
        reader = csv.reader(f)
        f.readline() # skip first row of column labels
        celebIDNames = dict()
        for row in reader:
            row = row[0].split('\t')
            celebIDNames[row[0]] = row[1] # assign celeb id to name
    return celebIDNames

def getVoiceListDict(celebIDNames):
    voiceList = list()
    for (root, dirs, files) in os.walk(DataPath + '/fbank', topdown=True):
        for currFile in files:
            rootLength = len(root.split('/'))
            celebID = root.split('/')[rootLength-2] #will only work for my computer, directory as given.
            # celebID = re.search(r'(?<=\/)(id[0-9]{5,5})(?=\/)',root).group(0)
            celebName = celebIDNames[celebID]
            if celebName.startswith(tuple(string.ascii_uppercase[5:])): #like authors, train only certain starting letters
                voiceList.append([os.path.join(root, currFile), celebName]) # list of lists (filepath,name)
    return voiceList

def getFaceListDict(celebIDNames):
    faceList = list()
    for (root, dirs, files) in os.walk(DataPath + '/VGG_ALL_FRONTAL', topdown=True):
        for currFile in files:
            rootLength = len(root.split('/'))
            celebName = root.split('/')[rootLength-1] #will only work for my computer, directory as given.
            if celebName.startswith(tuple(string.ascii_uppercase[5:])): #like authors, train only certain starting letters
                faceList.append([os.path.join(root, currFile), celebName]) # list of lists (filepath,name)
    return faceList

def getEnumerations(voiceListDict, faceListDict):
    commonNameSet1 = set()
    commonNameSet2 = set()
    for voiceNames in voiceListDict:
        commonNameSet1.add(voiceNames[1])
    for faceNames in faceListDict:
        commonNameSet2.add(faceNames[1])
    
    commonNameSet = set()
    commonNameSet = commonNameSet1 & commonNameSet2

    retVoiceList = list()
    retFaceList = list()
    for li in voiceListDict:
        if li[1] in commonNameSet:
            retVoiceList.append(li)
    for li in faceListDict:
        if li[1] in commonNameSet:
            retFaceList.append(li)
    
    commonNameList = sorted(commonNameSet) # return a list sorted alphabetically
    for x in retVoiceList:
        x.append(commonNameList.index(x[1])) # for each element in list, return its index/enumeration in commonNameList and append to that element
    for x in retFaceList:
        x.append(commonNameList.index(x[1]))
    
    return retVoiceList, retFaceList, len(commonNameList)

def train(epoch, voiceTrainLoader, faceTrainLoader, voiceNetwork, generatorNetwork, generatorNetworkOpt, discriminatorNetwork, discriminatorNetworkOpt, classifierNetwork, classifierNetworkOpt):
    
    trueLabels = torch.Tensor(128,1) #per 3.1.1, these labels indicate all faces in face is real
    trueLabels = trueLabels.new_ones((128,1))

    falseLabels = torch.Tensor(128,1) # all labels indiciate faces generated are fake
    falseLabels = falseLabels.new_zeros((128,1))


    loss = nn.BCELoss()
    lossC = nn.NLLLoss()
    voiceTrainLoader = iter(cycle(voiceTrainLoader)) # https://discuss.pytorch.org/t/in-what-condition-the-dataloader-would-raise-stopiteration/17483/2 to prevent stopiteration
    faceTrainLoader = iter(cycle(faceTrainLoader))

    for i in range(epoch):
        (voice, voiceLabel) = next(voiceTrainLoader)
        (face, faceLabel) = next(faceTrainLoader)
        voice = voice.cuda()
        voiceLabel = voiceLabel.cuda()
        face = face.cuda()
        faceLabel = faceLabel.cuda()
        trueLabels = trueLabels.cuda()
        falseLabels = falseLabels.cuda()

        generatorNetworkOpt.zero_grad()
        noise = torch.randn(128, 64, 1, 1)
        noise = noise.cuda()

        voiceEmbed = voiceNetwork(voice) # start with some voice embed with noise added
        generatedFake = generatorNetwork(voiceEmbed + noise)

        

        #Train generator
        genD = discriminatorNetwork(generatedFake) # real or fake classification for generator output
        genC = classifierNetwork(generatedFake) 
        genDLoss = loss(genD, trueLabels) # calculate loss for discrimin in relation to true face labels
        genCLoss = lossC(genC, voiceLabel) # calculate loss for classifier in relation to generated voice label

        genDLoss.backward(retain_graph=True) #backpropogate gradients
        genCLoss.backward()
        generatorNetworkOpt.step()




        #Train discriminator, need to skip every other iteration 
        
        # if ((i) % 2) == 0:
        discriminatorNetworkOpt.zero_grad()
        classifierNetworkOpt.zero_grad()

        trueDis = discriminatorNetwork(face) # pass true faces to discrim
        trueCla = classifierNetwork(face) # pass in true face to classifier
        trueDisLoss = loss(trueDis, trueLabels) # loss between discrminiator label decision and true label
        trueClaLoss = lossC(trueCla, faceLabel) # i loss between classifier output face identity to actual identity of face
        trueDisLoss.backward(retain_graph=True) # backpropogate gradients
        trueClaLoss.backward()

        genD = discriminatorNetwork(generatedFake.detach()) # detach weight of noise voice embed
        falseDisLoss = loss(genD, falseLabels) # loss between discriminator label deciison and false label
        falseDisLoss.backward()

        discriminatorNetworkOpt.step()
        classifierNetworkOpt.step()

        

        if i % 100 == 0:
            print("Epoch: %d, genDLoss: %f, falseDisLoss: %f, genCLoss: %f, trueDisLoss: %f, trueClaLoss: %f" % (i, genDLoss.item(), falseDisLoss.item(), genCLoss.item(), trueDisLoss.item(), trueClaLoss.item()))
        

################################################################################# author code
def rm_sil(voice_file, vad_obj):
    """
       This code snippet is basically taken from the repository
           'https://github.com/wiseman/py-webrtcvad'

       It removes the silence clips in a speech recording
    """
    audio, sample_rate = read_wave(voice_file)
    frames = frame_generator(20, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 20, 50, vad_obj, frames)

    if os.path.exists('tmp/'):
       shutil.rmtree('tmp/')
    os.makedirs('tmp/')

    wave_data = []
    for i, segment in enumerate(segments):
        segment_file = 'tmp/' + str(i) + '.wav'
        write_wave(segment_file, segment, sample_rate)
        wave_data.append(wavfile.read(segment_file)[1])
    shutil.rmtree('tmp/')

    if wave_data:
       vad_voice = np.concatenate(wave_data).astype('int16')
    return vad_voice

def test(generatorNetwork, voiceNetwork):
    generatorNetwork.eval()
    vad = webrtcvad.Vad(2)
    melFreq = MFCC(nfilt=64, lowerf=20., upperf=7200., samprate=16000, nfft=1024, wlen=0.025)

    for f in os.listdir("/home/oem/570Project/example_data"): #2 MODIFY
        if f.endswith(".wav"):
            fp = os.path.join("/home/oem/570Project/example_data", f) #3 MODIFY
            voice = rm_sil(fp, vad)
            # Extract log mel-spectrogra
            voice = melFreq.sig2logspec(voice).astype('float32')

            # Mean and variance normalization of each mel-frequency 
            voice = voice - voice.mean(axis=0)
            voice = voice / (voice.std(axis=0)+np.finfo(np.float32).eps)

            full_frame_number = 1000
            init_frame_number = voice.shape[0]
            while voice.shape[0] < full_frame_number:
                voice = np.append(voice, voice[0:init_frame_number], axis=0)
                voice = voice[0:full_frame_number,:]

            voice = voice.T[np.newaxis, ...]
            voice = torch.from_numpy(voice.astype('float32'))

            voice = voice.cuda()
            v = F.normalize(voiceNetwork(voice))
            face = generatorNetwork(v)

            vutils.save_image(face.detach().clamp(-1,1),
                      f.replace('.wav', '.png'), normalize=True)
################################################################################# author code


if __name__ == "__main__":
    # Loading data
    celebIDNames = getCelebID()
    voiceListDict = getVoiceListDict(celebIDNames)
    faceListDict = getFaceListDict(celebIDNames)
    (voiceList, faceList, classifierOutputChannel) = getEnumerations(voiceListDict, faceListDict) # this function gets all common voice/faces dataset

    # Custom dataset
    voiceDataset = voiceCustomDataset(voiceList)
    faceDataset = faceCustomDataset(faceList)

    voiceTrainLoader = torch.utils.data.DataLoader(voiceDataset, shuffle=True, drop_last=True,batch_size=128, num_workers=4)
    faceTrainLoader = torch.utils.data.DataLoader(faceDataset, shuffle=True, drop_last=True,batch_size=128,num_workers=4)

    #load networks
    DataPath = os.path.expanduser('/home/oem/570Project/pretrained_models/voice_embedding.pth') #4 MODIFY
    voiceNetwork = voiceEmbeddingNetwork()
    voiceNetwork = voiceNetwork.cuda()
    voiceNetwork.load_state_dict(torch.load(DataPath), strict=False)
    voiceNetwork.eval()

    generatorNetwork = generator()
    generatorNetwork = generatorNetwork.cuda()
    generatorNetwork = generatorNetwork.train()
    generatorNetworkOpt = optim.Adam(generatorNetwork.parameters(), lr=0.0002, betas=(0.5,0.999))

    discriminatorNetwork = discriminator()
    discriminatorNetwork = discriminatorNetwork.cuda()
    disrciminatorNetowrk = discriminatorNetwork.train()
    discriminatorNetworkOpt = optim.Adam(discriminatorNetwork.parameters(), lr=0.0002, betas=(0.5,0.999)) # decrease learning rate? D becoming too strong too fast, generator is not learning

    # discriminatorNetworkOpt = optim.Adam(discriminatorNetwork.parameters(), lr=0.000002, betas=(0.5,0.999)) # decrease learning rate? D becoming too strong too fast, generator is not learning

    classifierNetwork = classifier(classifierOutputChannel)
    classifierNetwork = classifierNetwork.cuda()
    classifierNetwork = classifierNetwork.train()
    classifierNetworkOpt = optim.Adam(classifierNetwork.parameters(), lr=0.0002, betas=(0.5,0.999))

    train(200, voiceTrainLoader, faceTrainLoader, voiceNetwork, generatorNetwork, generatorNetworkOpt, discriminatorNetwork, discriminatorNetworkOpt, classifierNetwork, classifierNetworkOpt)
    test(generatorNetwork, voiceNetwork)









