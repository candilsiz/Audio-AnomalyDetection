import os
import librosa
import soundfile as sf

def split_audio_files(input_directory, save_directory):
    file_counter = 0
    for file in os.listdir(input_directory):
        y, sr = librosa.load(os.path.join(input_directory, file), sr=None)
        half_duration_samples = len(y) // 2
        first_half = y[:half_duration_samples]
        second_half = y[half_duration_samples:]
        new_file1 = f"{file_counter:08d}.wav"
        new_file2 = f"{file_counter+1:08d}.wav"
        sf.write(os.path.join(save_directory, new_file1), first_half, sr)
        sf.write(os.path.join(save_directory, new_file2), second_half, sr)
        file_counter += 2

def load_audios(path, label):
    labels = list()
    audio_files = list()
    file_names = list()
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        audio, sample_rate = librosa.load(file_path, sr=None)
        audio_files.append(audio)
        labels.append(label)
        file_names.append(filename[5:]) # first 5 zeros are deleted
    return audio_files, labels, sample_rate, file_names
