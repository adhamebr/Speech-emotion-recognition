
#python3 /home/adham.ibrahim/speech_datasets/train/Embeddings.py
import os
# Change the current working directory to a different directory
os.chdir("/home/adham.ibrahim/speech_datasets/train")
import pandas as pd

# Specify the file path
file_path = '/home/adham.ibrahim/speech_datasets/train/train_df.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
print(df.head())

import os
import pandas as pd
import numpy as np


# data_speech = pd.DataFrame()
# data_speech['path'] = df['File_Path']
# data_speech['emotion'] = df['Emotion']

import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio

# # Load the pre-trained model and processor
model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

print(len(df))
# # Load your audio file as a PyTorch tensor

df['Embeddings'] = None  # Add a new column for embeddings
df['Embeddings'] = df['Embeddings'].astype(object)

embeddingsList = []
failed_loads = []

for i in range(len(df)):
    try:
        print(i)
        audio, sample_rate = torchaudio.load(df['File_Path'][i])

        # Preprocess the audio data
        input_value = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

        # Reshape input tensor to remove extra dimension
        input_values = input_value.squeeze(0)

        # Generate embeddings
        with torch.no_grad():
            embeddings = model(input_values).last_hidden_state.mean(dim=1)

        embeddingsList.append(embeddings)
        df.at[i, 'Embeddings'] = embeddings  # Assign embeddings to the DataFrame

        if i % 1000 == 0:
            print("Checkpoint", i)
            # Save the embeddingsList as .pt file
            filename_pt = f'/home/adham.ibrahim/speech_datasets/train/Embeddings_4/embeddings_{i}.pt'
            torch.save(embeddingsList, filename_pt)

            # Save the DataFrame with the embeddings column as .csv file
            filename_df = f'/home/adham.ibrahim/speech_datasets/train/Embeddings_4/data_speech_{i}.csv'
            df.to_csv(filename_df, index=False)

    except Exception as e:
        # If an exception occurs during torchaudio.load, add the index to the failed loads list
        print(f"Error loading audio at index {i}: {str(e)}")
        failed_loads.append(i)

# Save failed_loads as .txt file
filename_failed = f'/home/adham.ibrahim/speech_datasets/train/Embeddings_4/failed_loads.txt'
with open(filename_failed, 'w') as file:
    for index in failed_loads:
        file.write(f"{index}\n")

print("DONE embeddings")
print("Failed loads:", failed_loads)

print("DONE embeddings")


# Save the list of embeddings to a file
torch.save(embeddingsList, '/home/adham.ibrahim/speech_datasets/train/speech_embeddings_3.pt')
df.to_csv("/home/adham.ibrahim/speech_datasets/train/speech_embeddings_3.csv")

# Load the list of embeddings from the file
# loaded_embedding_list = torch.load('embedding_list.pth')

# # Print loaded embeddings
# for embedding in loaded_embedding_list:
#     print(embedding)

print("DONE") 
