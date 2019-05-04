'''
- Extract Frames from videos
- Extract Faces from frames
- Extract Audio from videos
- Split Audio Spectogram into Segments
- Stitch face from each timestamp with its corresponding spectogram piece
- Save as pickle
'''
#%%
import cv2
import os, pathlib
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import librosa
import librosa.display

#%%
# destination parent
data_dest_parent = pathlib.Path('.') / 'data'
data_dest_parent.mkdir(exist_ok=True)

# source of deepfake files
fakes_parent = pathlib.Path('/home/js8365/data/Sandbox/dataset-deepfakes/deepfaketimit/DeepfakeTIMIT/higher_quality')
fakes_dirs = [x for x in fakes_parent.iterdir() if x.is_dir()]

# source of real files
pristine_parent = pathlib.Path('/home/js8365/data/Sandbox/dataset-deepfakes/vidtimit')
pristine_dirs = [x for x in pristine_parent.iterdir() if x.is_dir()]
ignore_head = ['head', 'head2', 'head3', 'processed']

# destination for deepfake video frames
fakes_dest_frames = data_dest_parent / 'frames'
fakes_dest_frames.mkdir(exist_ok=True)

# destination for deepfake pickle aggregations
data_dest_pickles = data_dest_parent / 'picklejar'
data_dest_pickles.mkdir(exist_ok=True)

#%%
# loop through deepfake folders
for fakes_dir in fakes_dirs:
    files = list(fakes_dir.glob('*.avi'))
    for i, avi in enumerate(files):
        data = []

        wav_file = str(avi.stem).split('-')[0]
        wav_file_name = wav_file + '.wav'
        if(i == 0):
            # create an individual folder for each video file
            print("[Video] ", avi.stem)
            fakes_dest_frames_avi = fakes_dest_frames / str(avi.stem)
            fakes_dest_frames_avi.mkdir(exist_ok=True)

        cam = cv2.VideoCapture(str(avi))
        total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        print("\u2319 FPS: ", cam.get(5), " frames: ", total_frames)  #fps
        
        # wav_rate, wav = wavfile.read(fakes_dir / wav_file_name)
        wav, sr = librosa.load(fakes_dir / wav_file_name, sr=16000, mono=True)
        
        # reshape wav file content for each frame
        frame_blocks_sizes = int(np.shape(wav)[0]/total_frames)
        frame_wavs = np.reshape(wav[:frame_blocks_sizes*total_frames], (total_frames, frame_blocks_sizes))



        current_frame = 0
        while(True):
            ret, frame = cam.read()
            if ret:
                name = str(fakes_dest_frames_avi) + '/frame' + str(current_frame) + '.png'
                print('\u2319 frame ' + name)
                # cv2.imwrite(name, frame)
                print(np.shape(frame))
                dis_frame_wavs = frame_wavs[current_frame]
                

                D = np.abs(librosa.stft(dis_frame_wavs))**2
                # S = librosa.feature.melspectrogram(S=D)

                # plt.figure(figsize=(10, 4))
                # librosa.display.specshow(librosa.power_to_db(S,
                #                                             ref=np.max),
                #                             y_axis='mel', fmax=8000,
                #                             x_axis='time')
                # plt.colorbar(format='%+2.0f dB')
                # plt.title('Mel spectrogram')
                # plt.tight_layout()

                # mfccs = librosa.feature.mfcc(y=dis_frame_wavs, sr=sr, n_mfcc=40)
                # plt.figure(figsize=(10, 4))
                # librosa.display.specshow(mfccs, x_axis='time')
                # plt.colorbar()
                # plt.title('MFCC')
                # plt.tight_layout()
                
                data.append((frame, D, 1))
                current_frame += 1
                # break # remove this to continue

            else:
                break
        cam.release()
        pickle_out = open(str(data_dest_pickles) + '/' + str(avi.stem) + '.pickle', "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
        # cv2.destroyAllWindows()
        # break  # remove this to do all files
    # break # remove this to do all folders

#%%
# generate pristine video pickles
for pristine_dir in pristine_dirs:
    pristine_videos_dir = pristine_dir / 'video'
    pristine_audios_dir = pristine_dir / 'audio'
    pristine_video_dirs = [x for x in pristine_videos_dir.iterdir() if x.is_dir()]
    for pristine_video_dir in pristine_video_dirs:
        print("[Video] {}-{}".format(pristine_dir.stem, pristine_video_dir.stem))
        if pristine_video_dir.stem in ignore_head:
            continue
        video_frames = list(pristine_video_dir.glob('*.jpg'))  # file names come out of order
        total_frames = len(video_frames)
        wav_file_name = pristine_video_dir.stem + ".wav"
        wav, sr = librosa.load(pristine_audios_dir / wav_file_name, sr=16000, mono=True)
        
        # reshape wav file content for each frame
        frame_blocks_sizes = int(np.shape(wav)[0]/total_frames)
        frame_wavs = np.reshape(wav[:frame_blocks_sizes*total_frames], (total_frames, frame_blocks_sizes))

        # for each frame: read the frame pixels, find corresponding audio segment and run stft on the audio segment, package these two and a 0 into a tuple and feed into a data array (i know, all in one line, crazy)
        data = [(cv2.imread(str(x)),
                np.abs(librosa.stft(frame_wavs[int(video_frames[i].stem)-1]))**2,
                0) for i,x in enumerate(video_frames)]
        
        pickle_out = open(str(data_dest_pickles) + '/' + str(pristine_dir.stem) + '-' + str(pristine_video_dir.stem) + '.pickle', "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
        # break  # remove this to do all files
    # break  # remove this to do all folders
#%%
# # reading the pickle file
# pickle_in = open("data/picklejar/sx374-video-mstk0.pickle","rb")
# example_dict = pickle.load(pickle_in)
    