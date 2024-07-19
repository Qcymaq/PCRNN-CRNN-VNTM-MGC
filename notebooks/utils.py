import os
import librosa as lb
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from pydub import AudioSegment
type_list = {0: ["cailuong", "CaiLuong"], 1: ["catru", "Catru"], 2:["chauvan", "Chauvan"], 3: ["cheo", "Cheo"], 4: ["hatxam", "Xam"]}
def load_sample_directories(root, type_index, samples_list , num_of_samples, mode="random"):
    def padding(index):
        return str(index).zfill(3)  # Zero-pad the index to ensure three digits

    if mode == "random":
        # Load random samples
        random_indices = np.random.randint(0, 500, size=num_of_samples)
        for index in random_indices:
            dir_index = padding(index)
            dir_path = os.path.join(root, type_list[type_index][0], f"{type_list[type_index][1]}.{dir_index}.wav")
            samples_list[index] = {"dir": dir_path}

    elif mode == "all":
        # Load all samples
        for i in range(num_of_samples):
            dir_index = padding(i)
            dir_path = os.path.join(root, type_list[type_index][0], f"{type_list[type_index][1]}.{dir_index}.wav")
            samples_list[i] = {"dir": dir_path}

    return samples_list


def load_samples(samples_listdir):
    for index, sample in samples_listdir.items():
        try:
            file, sr = lb.load(sample["dir"])
            if len(samples_listdir[index]) == 1:  # Avoid adding multiple times
                samples_listdir[index]["sampling"] = file
        except FileNotFoundError:
            print(f"File not found: {sample['dir']}. Skipping...")
    return samples_listdir
def get_stft_samples(samples, n_fft=1024, hop_length=1723):
    for index, item in samples.items():
        if 'sampling' in item:
            stft_complex = lb.stft(item["sampling"], n_fft=n_fft, hop_length=hop_length)
            magnitude_stft = np.abs(stft_complex)
            log_magnitude_stft = np.log(magnitude_stft + 0.001)
            log_magnitude_stft = np.flipud(log_magnitude_stft)
            samples[index]["stft"] = log_magnitude_stft
    return samples


def plot_stft_sample(samples_stft, sample_index):
    if sample_index not in samples_stft:
        print(f"Sample index {sample_index} not found in the dictionary.")
        return

    stft_sample = samples_stft[sample_index]["stft"]

    plt.figure(figsize=(10, 6))
    plt.imshow(stft_sample, cmap='inferno', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'STFT of Sample {sample_index} (Cailuong)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


def get_log_mel_spectrogram(samples, type_index, sr=22050):
    for index, item in samples.items():
        if "sampling" in item:
            S = lb.feature.melspectrogram(y=item["sampling"], sr=sr)
            S_db = lb.amplitude_to_db(S, ref=np.max)
            samples[index]["mel-spec-db"] = S_db
    return samples
def plot_log_mel_spectrogram(sample):
    if "mel-spec-db" in sample:
        plt.figure(figsize=(10, 4))
        plt.imshow(sample["mel-spec-db"], cmap='inferno', origin='lower', aspect='auto')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Filterbank Frequencies')
        plt.tight_layout()
        plt.show()
    else:
        print("Log-mel-spectrogram not found in the sample.")



def split_audio_in_folder(input_folder, output_folder, segment_length_sec=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"): 
            input_audio_path = os.path.join(input_folder, filename)

            audio = AudioSegment.from_file(input_audio_path)

            segment_length_ms = segment_length_sec * 1000 
            num_segments = len(audio) // segment_length_ms

            for i in range(num_segments):
                start_time = i * segment_length_ms
                end_time = (i + 1) * segment_length_ms
                segment = audio[start_time:end_time]

                output_filename = f"{os.path.splitext(filename)[0]}_segment_{i}.wav"
                output_path = os.path.join(output_folder, output_filename)
                segment.export(output_path, format="wav")

def load_sample_directories_tensecs(root, type_index, samples_list , num_of_samples, mode="random"):
    if mode == "random":
        random_indices = np.random.randint(0, 1500, size=num_of_samples)
        for i in random_indices:
            index = i // 3
            segment_number = i % 3
            dir_path = os.path.join(root, type_list[type_index][0], f"{type_list[type_index][1]}.{index:03d}_segment_{segment_number}.wav")
            samples_list[i] = {"dir": dir_path}

    elif mode == "all":
        for i in range(num_of_samples):
            # Calculate index and segment number
            index = i // 3
            segment_number = i % 3
            # Change file name accordingly
            dir_path = os.path.join(root, type_list[type_index][0], f"{type_list[type_index][1]}.{index:03d}_segment_{segment_number}.wav")
            samples_list[i] = {"dir": dir_path}

    return samples_list
def load_samples_tensecs(samples_listdir):
    for index, sample in samples_listdir.items():
        try:
            if "sampling" not in sample:  # Check if sampling already exists
                file, sr = lb.load(sample["dir"])
                samples_listdir[index]["sampling"] = file
        except FileNotFoundError:
            print(f"File not found: {sample['dir']}. Skipping...")
    return samples_listdir

def get_stft(file_path, n_fft=1024, hop_length=512, n_mels=128, time_frames=513):
    signal, sr = lb.load(file_path, sr=None)
    stft_matrix = lb.stft(signal, n_fft=n_fft, hop_length=hop_length)

    magnitude_stft = np.abs(stft_matrix)
    log_magnitude_stft = np.log(magnitude_stft + 0.001)
    log_magnitude_stft = np.flipud(log_magnitude_stft)

    # Reshape the STFT to the desired shape
    padded_stft = np.zeros((time_frames, n_mels))
    padded_stft[:log_magnitude_stft.shape[0], :log_magnitude_stft.shape[1]] = log_magnitude_stft[:time_frames, :n_mels]

    # Create image with color channels (magnitude as red, phase as green, zeros as blue)
    image = np.stack((padded_stft, padded_stft, np.zeros_like(padded_stft)), axis=-1)

    image = np.array(image)
    padded_stft= np.array(padded_stft)
    return image, padded_stft

def process_audio_files(audio_dir):
    # List to store features and corresponding labels
    stft_img =[]
    stft_list = []
    labels = []

    # Iterate over the subdirectories (genres)
    for genre_folder in os.listdir(audio_dir):
        genre_path = os.path.join(audio_dir, genre_folder)
        if os.path.isdir(genre_path): 
            # Iterate over all audio files in the genre directory
            for filename in os.listdir(genre_path):
                if filename.endswith('.wav'):  # Assuming all audio files are in .wav format
                    file_path = os.path.join(genre_path, filename)

                    img, stft = get_stft(file_path)
                    # Append features to data lists
                    stft_list.append(stft)
                    stft_img.append(img)

                    # Append label to labels list (genre folder name)
                    labels.append(genre_folder)

    # Convert lists to NumPy arrays

    labels = np.array(labels)

    return stft_img, stft_list, labels

def compute_melgram(audio_path):
    # Mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  

    # Load audio file
    src, sr = lb.load(audio_path, sr=SR)

    # Ensure the audio is of desired duration
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)
    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]

    # Compute the mel-spectrogram
    melgram = lb.feature.melspectrogram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS)

    # Convert to log scale
    log_melgram = lb.power_to_db(melgram ** 2, ref=np.max)

    log_melgram = log_melgram[np.newaxis, : ]

    return log_melgram

def process_audio_files_crnn(audio_dir):
    data = []
    labels = []

    # Iterate over the subdirectories (genres)
    for genre_folder in os.listdir(audio_dir):
        genre_path = os.path.join(audio_dir, genre_folder)
        if os.path.isdir(genre_path):  
            for filename in os.listdir(genre_path):
                if filename.endswith('.wav'):  # Assuming all audio files are in .wav format
                    file_path = os.path.join(genre_path, filename)

                    # Compute log-mel spectrogram for the current audio file
                    log_mel_spectrogram = compute_melgram(file_path)

                    # Append log-mel spectrogram to data list
                    data.append(log_mel_spectrogram)

                    # Append label to labels list (genre folder name)
                    labels.append(genre_folder)

    return data, labels