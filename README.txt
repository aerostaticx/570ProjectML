README
1) This experiment is meant to be run on Linux machines with CUDA capability and Conda. There are several packages to install and datasets to download.
2) Please download both datasets here:

Voxceleb: audio files (npy format) of celebrities speaking
https://drive.google.com/file/d/1T5Mv_7FC2ZfrjQu17Rn9E24IOgdii4tj/view

VGGFace: face images of celebrities in different conditions
https://drive.google.com/file/d/1T5Mv_7FC2ZfrjQu17Rn9E24IOgdii4tj/view

3) Extract and place the folders fbank and VGG_ALL_FRONTAL into 570Project folder.
4) Run ./install.sh in 570Project
5) Download webrtcvad from https://drive.google.com/file/d/1FYICqyEh1d45TUksibduBK30h1AjGUh8/view
6) Extract and place all files in 570Project, confirm any rewrites
7) Type and execute 'source activate voice2face' in the terminal.
8) Navigate to 570Project and run implementation.py with 'python implemenentation.py' or IDE of your choice
9) Png results will be deposited in the 570Project folder.


**Code reused from authors has been marked clearly with ##### commented sections in implementation.py. These are mostly image/audio file conversions, sound processing and code from other repositories that the author's themselves reused. All other sections are original code.

Links to reused are here:
https://github.com/cmu-mlsp/reconstructing_faces_from_voices/blob/master/utils.py
https://github.com/cmu-mlsp/reconstructing_faces_from_voices/blob/master/utils.py
https://github.com/cmu-mlsp/reconstructing_faces_from_voices/blob/master/gan_test.py

mfcc.py and vad.py are required files for certain necessary packages. They are taken from here:
https://github.com/cmu-mlsp/reconstructing_faces_from_voices/blob/master/mfcc.py
https://github.com/cmu-mlsp/reconstructing_faces_from_voices/blob/master/vad.py


