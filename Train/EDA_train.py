# python3 /home/adham.ibrahim/speech_datasets/train/EDA_train.py
import os

# Change the current working directory to a different directory
os.chdir("/home/adham.ibrahim/speech_datasets/train")
import os
import pandas as pd

# Specify the directory path
directory_path = '/home/adham.ibrahim/speech_datasets/train'

# Initialize an empty dictionary to store the DataFrames
data_dict = {}

# Iterate through subdirectories
for folder_name in os.listdir(directory_path):
    folder_path = os.path.join(directory_path, folder_name)
    
    # Check if the item in the directory is a subdirectory
    if os.path.isdir(folder_path):
        
        # Construct the expected CSV file path (assuming the CSV file has the same name as the folder)
        csv_file_path = os.path.join(folder_path, f'{folder_name}.csv')
      
        # Check if the CSV file with the same name as the folder exists
        if os.path.exists(csv_file_path):
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path)
            
            # Store the DataFrame in the dictionary with the folder name as the key
            data_dict[folder_name] = df

# Now, data_dict contains DataFrames for each subdirectory where the keys are the subdirectory names
# Iterate through the data_dict and make the "Emotion" column values lowercase
for folder_name, df in data_dict.items():
    if "Emotion" in df.columns:
        df["Emotion"] = df["Emotion"].str.lower()

# Iterate through the data_dict and make the "Emotion" column values lowercase
for folder_name, df in data_dict.items():
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].str.lower()

# Define a mapping of values to be replaced
emotion_mapping = {
    'neutral': 'neu',
    'angry': 'ang',
    'happy': 'hap',
    'surprised': 'sur',
    'surprise': 'sur',
    'disgust': 'dis',
    'disgusted': 'dis',
    'fear': 'fea',
    'fearful': 'fea',
    'excited' : 'exc'
}

# Iterate through the data_dict and apply the mapping to the "Emotion" column
for folder_name, df in data_dict.items():
    if "Emotion" in df.columns:
        df["Emotion"] = df["Emotion"].replace(emotion_mapping)
import librosa
import soundfile

dataset = ['Emotion_Speech_Dataset', 'emov_db', 'MELD.Raw', 'savee', 'CREMA-D', 'ravdess', 'IEMOCAP_full_release', 'JL_corpus']
def change_sample_rate(file_path, target_sample_rate=16000):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)

        # Check if the sample rate needs to be changed
        if sr != target_sample_rate:
            # Resample the audio
            audio_resampled = librosa.resample(y = audio, orig_sr = sr, target_sr= target_sample_rate)

            # Save the resampled audio
            soundfile.write(file_path, audio_resampled, target_sample_rate)

            print(f"Sample rate changed to {target_sample_rate} Hz for file: {file_path}")
        # else:
        #     print(f"Sample rate is already {target_sample_rate} Hz for file: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Assuming data_dict is your dictionary and dataset is a list of indices
# For example: dataset = [1, 2, 3, ...]
for i in range(len(dataset)):
    file_paths = data_dict[dataset[i]]['File_Path']
    for file_path in file_paths:
        change_sample_rate(file_path)
        
print("All samples are 16khz")
