import pandas as pd 
import os, shutil
import numpy as np

#set the index of data and transfer to the pd 
df = pd.read_excel('local/speaker_data.xlsx', index=['id'])
dic = df.set_index('id').T.to_dict('list')

audio_dir = '/data/act19yz/kaldi/egs/ADSR/audio'

audio_kind = 'mix'  #if audio_kind = ctlall is means the control data to be training data, and ctl data to be testing data
#if audio_kind =ctl is means the control data as training data and dysarthric speech data to be testing data
#if audio_kind =dys is means the dysarthric data as training data and dysarthric speech data as testing data
#and if audio_kind = mix is means the control data and dysarthric speech data will mix into training data

#Delete the bad file and the errors.npy 
errors = np.load('local/errors.npy')
print("the number of error files:",len(errors))
print("the errors file are that:")
print(errors)

for subdir, dirs, files in os.walk(audio_dir):
    for file in files:
        filepath = subdir + os.sep + file
        if file in errors:
            print("removing bad files --> ", file)
            os.remove(filepath)


print("Creating wav.scp, text and utt2spk files for train set, please wait..")

# Get train files
#all data are dysarthric and control mix data
if audio_kind == 'mix':
    # Get train files
    wav_file = open("wav.scp", "w")
    text_file = open("text", "w")
    utt_file = open("utt2spk", "w")
    #training data
    for subdir, dirs, files in os.walk(audio_dir):
        for file in files:
            filepath = subdir + os.sep + file
            #two blocks' dysarthric files in the training data
            if filepath.endswith('.wav') and file.startswith('C'):
                #separate the blocks into three blocks B2 is testing data
                for i in dic:
                    if i in file:
                        name = file[:-4]
                        word = str(dic[i]).strip('[]').strip("''").upper()
                        spk = file.split('_')[0] 
                            
                        wav_file.write(name + " "+ filepath + "\n")
                        text_file.write(name + " " + word + "\n")
                        utt_file.write(name + " " + spk + "\n")
            #three blocks
            elif 'B1' in file or 'B3' in file and 'B2' not in file:
                for i in dic:
                    if i in file:
                        name = file[:-4]
                        word = str(dic[i]).strip('[]').strip("''").upper()
                        spk = file.split('_')[0] 
                        
                        wav_file.write(name + " "+ filepath + "\n")
                        text_file.write(name + " " + word + "\n")
                        utt_file.write(name + " " + spk + "\n")

    wav_file.close()
    text_file.close()
    utt_file.close()
    print("Finished creating train files! by dys ways: dysarthric data is be training data")

    my_files = ['wav.scp', 'text', 'utt2spk']
    for idx in my_files:
        #baseline
        shutil.move(idx, 'audio/train_mix_pitch/')
        #plus the delta  training
        #shutil.move(idx, 'audio/train_plusdelta_mix/')
    print(" ")
    print("....")
    print("Creating test files....")
    #testing data
    wav_file = open("wav.scp", "w")
    text_file = open("text", "w")
    utt_file = open("utt2spk", "w")
    for subdir, dirs, files in os.walk(audio_dir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith('.wav') and file.startswith('C') ==False and 'B2' in file:
            # if filepath.endswith('.wav') and file.startswith('C'):
                for i in dic:
                    if i in file:
                        name = file[:-4]
                        word = str(dic[i]).strip('[]').strip("''").upper()
                        spk = file.split('_')[0] 
                                
                        wav_file.write(name + " "+ filepath + "\n")
                        text_file.write(name + " " + word + "\n")
                        utt_file.write(name + " " + spk + "\n")


    wav_file.close()
    text_file.close()
    utt_file.close()

    for idx in my_files:
        #baseling
    	shutil.move(idx, 'audio/test_mix_pitch/')
        #plus delta training
        #shutil.move(idx, 'audio/test_plusdelta_mix/')
    print("finished creating all files! by ctlall ways:dysarthric data is be testing data!! ")
# if audio_kind == 'ctl':
#     # Get train files
#     wav_file = open("wav.scp", "w")
#     text_file = open("text", "w")
#     utt_file = open("utt2spk", "w")
#     for subdir, dirs, files in os.walk(audio_dir):
#         for file in files:
#             filepath = subdir + os.sep + file
#             #all control files in the training data
#             if filepath.endswith('.wav') and file.startswith('C'):
#                 for i in dic:
#                     if i in file:
#                         name = file[:-4]
#                         word = str(dic[i]).strip('[]').strip("''").upper()
#                         spk = file.split('_')[0] 
                        
#                         wav_file.write(name + " "+ filepath + "\n")
#                         text_file.write(name + " " + word + "\n")
#                         utt_file.write(name + " " + spk + "\n")
#             # #three blocks
#             # elif 'B1' in file or 'B3' in file and 'B2' not in file:
#             #     for i in dic:
#             #         if i in file:
#             #             name = file[:-4]
#             #             word = str(dic[i]).strip('[]').strip("''").upper()
#             #             spk = file.split('_')[0] 
                        
#             #             wav_file.write(name + " "+ filepath + "\n")
#             #             text_file.write(name + " " + word + "\n")
#             #             utt_file.write(name + " " + spk + "\n")

#     wav_file.close()
#     text_file.close()
#     utt_file.close()
#     print("Finished creating train files! by ctl ways: control data is be training data")

#     my_files = ['wav.scp', 'text', 'utt2spk']
#     for idx in my_files:
#         shutil.move(idx, 'audio/train_ctl')

#     print(" ")
#     print("....")
#     print("Creating test files....")

#     wav_file = open("wav.scp", "w")
#     text_file = open("text", "w")
#     utt_file = open("utt2spk", "w")
#     for subdir, dirs, files in os.walk(audio_dir):
#         for file in files:
#             filepath = subdir + os.sep + file
#             if filepath.endswith('.wav') and file.startswith('C') ==False and 'B2' in file:
#                 for i in dic:
#                     if i in file:
#                         name = file[:-4]
#                         word = str(dic[i]).strip('[]').strip("''").upper()
#                         spk = file.split('_')[0] 
                        
#                         wav_file.write(name + " "+ filepath + "\n")
#                         text_file.write(name + " " + word + "\n")
#                         utt_file.write(name + " " + spk + "\n")


#     wav_file.close()
#     text_file.close()
#     utt_file.close()
#     for idx in my_files:
#         shutil.move(idx, 'audio/test_ctl/')
#     print("finished creating all files! by ctl ways: dysarthric data is testing data")





# #all data are control data
# elif audio_kind = 'ctlall':
#     # Get train files
#     wav_file = open("wav.scp", "w")
#     text_file = open("text", "w")
#     utt_file = open("utt2spk", "w")
#     #training data
#     for subdir, dirs, files in os.walk(audio_dir):
#         for file in files:
#             filepath = subdir + os.sep + file
#             #all control files in the training data
#             if filepath.endswith('.wav') and file.startswith('C'):
#                 #separate the blocks into three blocks B2 is testing data
#                 if 'B1' in file or 'B3' in file and 'B2' not in file:
#                     for i in dic:
#                         if i in file:
#                             name = file[:-4]
#                             word = str(dic[i]).strip('[]').strip("''").upper()
#                             spk = file.split('_')[0] 
                            
#                             wav_file.write(name + " "+ filepath + "\n")
#                             text_file.write(name + " " + word + "\n")
#                             utt_file.write(name + " " + spk + "\n")
#             #three blocks
#             # elif 'B1' in file or 'B3' in file and 'B2' not in file:
#             #     for i in dic:
#             #         if i in file:
#             #             name = file[:-4]
#             #             word = str(dic[i]).strip('[]').strip("''").upper()
#             #             spk = file.split('_')[0] 
                        
#             #             wav_file.write(name + " "+ filepath + "\n")
#             #             text_file.write(name + " " + word + "\n")
#             #             utt_file.write(name + " " + spk + "\n")

#     wav_file.close()
#     text_file.close()
#     utt_file.close()
#     print("Finished creating train files! by ctlall ways: control data is be training data")

#     my_files = ['wav.scp', 'text', 'utt2spk']
#     for idx in my_files:
#         shutil.move(idx, 'audio/train_ctlall/')

#     print(" ")
#     print("....")
#     print("Creating test files....")
#     #testing data
#     wav_file = open("wav.scp", "w")
#     text_file = open("text", "w")
#     utt_file = open("utt2spk", "w")
#     for subdir, dirs, files in os.walk(audio_dir):
#         for file in files:
#             filepath = subdir + os.sep + file
#             # if filepath.endswith('.wav') and file.startswith('C') ==False and 'B2' in file:
#             if filepath.endswith('.wav') and file.startswith('C'):
#                 if 'B2' in file:
#                     for i in dic:
#                         if i in file:
#                             name = file[:-4]
#                             word = str(dic[i]).strip('[]').strip("''").upper()
#                             spk = file.split('_')[0] 
                            
#                             wav_file.write(name + " "+ filepath + "\n")
#                             text_file.write(name + " " + word + "\n")
#                             utt_file.write(name + " " + spk + "\n")


#     wav_file.close()
#     text_file.close()
#     utt_file.close()

#     for idx in my_files:
#         shutil.move(idx, 'audio/test_ctlall/')
#     print("finished creating all files! by ctlall ways:control data is be testing data and all data is ctl data!! ")





# #all data are dysarthric data
# elif audio_kind = 'dys':
#     # Get train files
#     wav_file = open("wav.scp", "w")
#     text_file = open("text", "w")
#     utt_file = open("utt2spk", "w")
#     #training data
#     for subdir, dirs, files in os.walk(audio_dir):
#         for file in files:
#             filepath = subdir + os.sep + file
#             #two blocks' dysarthric files in the training data
#             if filepath.endswith('.wav') and file.startswith('C') == False:
#                 #separate the blocks into three blocks B2 is testing data
#                 if 'B1' in file or 'B3' in file and 'B2' not in file:
#                     for i in dic:
#                         if i in file:
#                             name = file[:-4]
#                             word = str(dic[i]).strip('[]').strip("''").upper()
#                             spk = file.split('_')[0] 
                            
#                             wav_file.write(name + " "+ filepath + "\n")
#                             text_file.write(name + " " + word + "\n")
#                             utt_file.write(name + " " + spk + "\n")
#             #three blocks
#             # elif 'B1' in file or 'B3' in file and 'B2' not in file:
#             #     for i in dic:
#             #         if i in file:
#             #             name = file[:-4]
#             #             word = str(dic[i]).strip('[]').strip("''").upper()
#             #             spk = file.split('_')[0] 
                        
#             #             wav_file.write(name + " "+ filepath + "\n")
#             #             text_file.write(name + " " + word + "\n")
#             #             utt_file.write(name + " " + spk + "\n")

#     wav_file.close()
#     text_file.close()
#     utt_file.close()
#     print("Finished creating train files! by dys ways: dysarthric data is be training data")

#     my_files = ['wav.scp', 'text', 'utt2spk']
#     for idx in my_files:
#         shutil.move(idx, 'audio/train_dys/')

#     print(" ")
#     print("....")
#     print("Creating test files....")
#     #testing data
#     wav_file = open("wav.scp", "w")
#     text_file = open("text", "w")
#     utt_file = open("utt2spk", "w")
#     for subdir, dirs, files in os.walk(audio_dir):
#         for file in files:
#             filepath = subdir + os.sep + file
#             if filepath.endswith('.wav') and file.startswith('C') ==False:
#             # if filepath.endswith('.wav') and file.startswith('C'):
#                 if 'B2' in file:
#                     for i in dic:
#                         if i in file:
#                             name = file[:-4]
#                             word = str(dic[i]).strip('[]').strip("''").upper()
#                             spk = file.split('_')[0] 
                                
#                             wav_file.write(name + " "+ filepath + "\n")
#                             text_file.write(name + " " + word + "\n")
#                             utt_file.write(name + " " + spk + "\n")


#     wav_file.close()
#     text_file.close()
#     utt_file.close()

#     for idx in my_files:
#         shutil.move(idx, 'audio/test_dys/')
#     print("finished creating all files! by ctlall ways:dysarthric data is be testing data and all data is dys data!! ")


