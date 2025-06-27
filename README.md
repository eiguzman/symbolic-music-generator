# Symbolic Music Generation with Variational Autoencoders
-Edgar Guzman

*Originally a final assignment for CSE153 at the University of California, San Diego*

**Task**: *Make some beautiful music!*

This project develops deep learning frameworks to generate symbolic music in both conditioned and unconditioned modes. Trained on thousands of copyright-free MIDI files, the goal is to develop structurally sound and audibly appealing piano compositions. The datasets used in this project are available as zip files from [PDMX](https://pnlong.github.io/PDMX.demo/), [MAESTRO](https://magenta.tensorflow.org/datasets/maestro#v300), and [MIDI Chords](https://github.com/ldrolez/free-midi-chords).

# Task 1a - 1D Variational Autoencoder
### Introduction 
Our first model aims to generate music using symbolic, unconditioned generation. We train a model on over roughly 2,500 single-instrument, double-staffed MIDI files to determine both note and duration distributions. All of these MIDI files were sampled from PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing. This dataset contains over 250,000 public domain files for use in ML training.

### Data Cleaning
Processing the data required the use of multiple music packages. Initial attempts used <code>MidiUtil</code> and <code>Mido</code>, but since the model requires the use of a piano roll, <code>pretty_midi</code> was used for accessibility. While most MIDI files were cleaned up and ready for use, many had errors that pretty_midi was unable to solve. For example, a few files had a time scale of 128/4 that appeared as zeros. Any file that threw out an error was ignored in our training model.

```python
import os
import time
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim

from glob import glob
from torch.utils.data import Dataset, DataLoader
 ```

### Model Description
The algorithm I chose to employ was a Variational Autoencoder. The way this model works is best described as an hourglass-shaped neural network. This model works by taking as input a 2-channel piano roll of 128 possible pitches, multiplied by the number of notes we want to generate. Each note, pitch, and channel combination is called a node. The encoder goes through multiple layers, each with fewer nodes. These nodes capture key features that would be difficult to discover through manual feature extraction. Additionally, these nodes continue to decrease until they reach the latent dimension, the smallest layer with the most important features. The layers again increase until the output layer's size is the same as the input's shape. In short, **VAEs describe a probability distribution over the latent variables, whose encoder outputs the mean and variance of the distribution that is used to sample from the distribution and generate new notes.**

The goal of using a VAE is to effectively learn patterns from multiple music samples. The result of running this algorithm is a set of features, or model weights, that best represent the sample files as a whole. Through each iteration of the model, a loss function calculates how well the model is related to a batch of randomly sampled training files. Ideally, the loss decreases with each epoch. Realistically, the loss will be very high, as each musical piece varies greatly through many features. A low loss usually means that the model is going to predict dissonant chords in an attempt to fill the most used notes positions. We will discuss methods used to prevent this from happening. We want to optimize our weights to produce a structurally sound song without generating a song that sounds slightly similar to every song it was trained on.

### Parameters, Arguments, and Variables
The following parameters and their selected arguments are described below:

| Parameter | Description | Argument |
| --------- | ----------- | -------- |
| `seq_length` | Number of generated notes / Number of samples | 400 |
| `input_dim` | Input dimension shape | 2 * 128 * `seq_length` |
| `hidden_dim` | Encoder hidden dimension shape | 2048 |
| `latent_dim` | Encoder latent dimension shape | 128 |
| `output_dum` | Output dimension shape | `input_dim` |
| `root_pitch` | Default pitch for generating notes | 60 |
| `batch_size` | Number of samples processed in one iteration | 250 |
| `num_samlpes` | Number of MIDI files to generate | 1 `or` 3 |
| `max_notes_per_time` | Maximum chord size | 3 |

This model also uses the following variable values:

| Variable | Description | Value |
| -------- | ----------- | ----- |
| `max_duration` | Maximum note duration allowed | 50 |
| `num_epochs` | Number of epochs in the loss function | 2 |
| `threshold` | Probability threshold to qualify for candidacy | 0.035 |
| `max_history` | Maximum number of repeated notes | 4 |
| `offset` | Pitch shift change | 0 `or` 2 |
| `note_duration` | Default note duration | .25 (eighth note) |


### Code
```python
# 1. Data Preparation: Convert MIDI to a sequence representation with note durations
def midi_to_sequence(midi_path, seq_length):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        # Return empty or dummy data if load fails
        piano_roll = np.zeros((128, seq_length))
        duration_array = np.zeros((128, seq_length))
        return piano_roll, duration_array
    piano_roll = pm.get_piano_roll(fs=20)  # 20 frames per second
    # Binarize piano roll
    piano_roll = (piano_roll > 0).astype(np.float32)
    # Generate note duration information
    duration_array = np.zeros_like(piano_roll)

    for pitch in range(128):
        pitch_vector = piano_roll[pitch, :]
        diff = np.diff(pitch_vector, prepend=0)
        onsets = np.where(diff == 1)[0]
        offsets = np.where(diff == -1)[0]
        if len(offsets) < len(onsets):
            offsets = np.append(offsets, len(pitch_vector))
        for onset, offset in zip(onsets, offsets):
            duration = offset - onset
            duration_array[pitch, onset:offset] = duration
            
    # Pad or truncate to fixed length
    if piano_roll.shape[1] < seq_length:
        pad_width = seq_length - piano_roll.shape[1]
        piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)), mode='constant')
        duration_array = np.pad(duration_array, ((0, 0), (0, pad_width)), mode='constant')
    else:
        piano_roll = piano_roll[:, :seq_length]
        duration_array = duration_array[:, :seq_length]

    return piano_roll, duration_array 
```

```python
# 2. Dataset class
class MidiDataset(Dataset):
    def __init__(self, midi_files, seq_length=400):
        self.files = midi_files
        self.seq_length = seq_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        midi_path = self.files[idx]
        piano_roll, duration_array = midi_to_sequence(midi_path, self.seq_length)
        max_duration = 50
        duration_norm = np.clip(duration_array / max_duration, 0, 1)
        # Stack piano roll and duration as channels
        # Shape: (2, pitch, time)
        sample = np.stack([piano_roll, duration_norm], axis=0)
        return torch.tensor(sample, dtype=torch.float32)
```

```python
# 3. Define VAE components with dual outputs
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_length=400):
        super().__init__()
        self.seq_length = seq_length
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.output_dim = output_dim
        # Split into two heads: one for pitch, one for duration
        self.fc_pitch = nn.Linear(hidden_dim, 128 * seq_length)  # pitch output
        self.fc_duration = nn.Linear(hidden_dim, 128 * seq_length)  # duration output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        pitch_logits = self.fc_pitch(h)
        duration_logits = self.fc_duration(h)
        # reshape to (batch, channels=2, pitch=128, time=seq_length)
        pitch_logits = pitch_logits.view(-1, 128, self.seq_length)
        duration_logits = duration_logits.view(-1, 128, self.seq_length)
        # Sigmoid for pitch (binary presence)
        pitch_probs = self.sigmoid(pitch_logits)
        return pitch_probs, duration_logits

class MusicVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        pitch_probs, duration_logits = self.decoder(z)
        return pitch_probs, duration_logits, mu, logvar
```

```python
# 4. Loss function with dual components
def loss_function(pitch_probs, duration_logits, x, mu, logvar):
    # Split x into target pitch and duration
    target_pitch = x[:, 0, :, :]  # shape: (batch, 128, time)
    target_duration = x[:, 1, :, :]  # shape: (batch, 128, time)
    max_duration = 50

    # Compute binary cross-entropy for pitch
    BCE = nn.functional.binary_cross_entropy(pitch_probs, target_pitch, reduction='sum')
    # Compute MSE for durations (regression)
    duration_pred = duration_logits
    duration_target = target_duration
    # Denormalize durations for loss calculation if desired
    MSE = nn.functional.mse_loss(duration_pred, duration_target, reduction='sum')
    # KLD for VAE
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + MSE + KLD

C_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
def pitch_shift_to_c_major(pitch, base_pitch=60):
    pitch_in_scale = pitch - base_pitch
    # Find the closest scale step
    distances = [abs(pitch_in_scale - interval) for interval in C_MAJOR_SCALE]
    min_index = np.argmin(distances)
    shifted_pitch = base_pitch + C_MAJOR_SCALE[min_index]
    return shifted_pitch

CHORD_PATTERNS = [
    [0, 4, 7],       # Major triad
    [-3, 0, 4],      # Minor triad1
    [-7, -3, 0],     # Minor triad2
    [-5, -1, 2]      # Other chord
]
def get_custom_chord(root_pitch):
    # Randomly select one of your custom chord patterns
    pattern = CHORD_PATTERNS[np.random.randint(len(CHORD_PATTERNS))]
    # Transpose pattern to the root pitch
    chord_pitches = [root_pitch + interval for interval in pattern]
    # Keep within MIDI pitch range
    chord_pitches = [p for p in chord_pitches if 0 <= p <= 127]
    return chord_pitches
```

```python
# 5. Setup training
midi_files = (glob("mid/0/0/*.mid") +
              glob("mid/0/1/*.mid") + 
              glob("mid/0/2/*.mid") + 
              glob("mid/0/3/*.mid") + 
              glob("mid/0/4/*.mid"))
dataset = MidiDataset(midi_files, seq_length=400)
dataloader = DataLoader(dataset, batch_size=250, shuffle=True, num_workers=0)

input_dim = 2 * 128 * 400  # 2 channels: piano roll + duration
hidden_dim = 2048
latent_dim = 128

model = MusicVAE(input_dim, hidden_dim, latent_dim, seq_length=400)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 6. Training loop
print("Start Training")
t1 = time.time()
num_epochs = 2
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        batch = batch.view(batch.size(0), -1)  # flatten to (batch, input_dim)
        optimizer.zero_grad()
        pitch_probs, duration_logits, mu, logvar = model(batch)
        loss = loss_function(pitch_probs, duration_logits, batch.view(batch.size(0), 2, 128, -1), mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
t2 = time.time()
print(f"Finished Training in {round(t2 - t1, 2)}s")
```

### Music Generation
Our model then generates notes, either single notes or chords, based on the probability of that note being played. The way this works is simple: the VAE determines a probability distribution of every possible note that was played within a certain beat. The algorithm filters out any note with a probability below the threshold value. Then, it checks the length of this array. If there are more than three notes, there is a high probability of generating a dissonant chord; therefore, the three notes with the greatest probability are selected. If there are 1 or 2 notes, randomly select one. If no notes were above the threshold, generate the previous note. This is important so as to reduce randomness, increase tone harmonics, and produce a more audibly pleasing composition. Other constraints, such as limiting repeating notes to 4 consecutive notes, are also used to reduce noise and improve our generation capabilities. 

One advantage of implementing the model this way is that it prevents bias in our model. By limiting how often the most popular or frequent note is played, our model has a reduced risk of predicting the same note for all possible positions. A disadvantage of this method is that more advanced models that can benefit from larger datasets are available, but due to our limited time and memory constraints, a simpler model was chosen. Additionally, this model heavily predicts chords over single notes. The reasons why this happens are twofold. First, a low threshold value means that more candidate notes are chosen, so chords are more likely to be played. Second, this is due to the training dataset containing a significantly large number of chords, making single note generation more difficult.

```python
# 7. Generate new music with note durations
def generate_music_with_repetition_control(model, latent_dim, num_samples=1, seq_length=400, max_notes_per_time=3):
    threshold= 0.035
    max_history = 4  # number of previous notes to compare
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        pitch_probs, duration_logits = model.decoder(z)
        pitch_probs = pitch_probs.cpu().numpy()
        duration_logits = duration_logits.cpu().numpy()
    
        for i in range(num_samples):
            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            dur_pred = duration_logits[i]
            dur_pred = np.clip(dur_pred, 0, 1)
    
            previous_notes = [[60]]

            start_time = 0
            for t in range(seq_length):
                selected_pitches = []
                pitch_probs_t = pitch_probs[i, :, t]
                pitch_probs_t = np.array(pitch_probs_t) / sum(pitch_probs_t)
                candidate_pitches = np.where(pitch_probs_t > threshold)[0].tolist()
                if len(candidate_pitches) == 0:
                    last_pitch = previous_notes[-1][-1]
                    shifted_pitch = pitch_shift_to_c_major(last_pitch)
                    selected_pitches.append(shifted_pitch)
                elif len(candidate_pitches) > max_notes_per_time:
                    # Pick a root pitch from candidates
                    root_pitch = np.random.choice(candidate_pitches)
                    # Generate the major chord pitches
                    chord_pitches = get_custom_chord(root_pitch)
                    # Assign the chord pitches as the selected notes
                    selected_pitches.extend(chord_pitches)
                else:
                    choose = np.random.choice(candidate_pitches)
                    shifted_pitch = pitch_shift_to_c_major(choose)
                    selected_pitches.append(shifted_pitch)

                offset = 0
                if all(selected_pitches == i for i in previous_notes):
                    offset = 2
                for pitch_lst in selected_pitches:
                    if type(pitch_lst) != list:
                        pitch_lst = [pitch_lst]
                    for p in pitch_lst:
                        note_duration = dur_pred[p, t]
                        if note_duration == 0:
                            note_duration = .25
                        end_time = start_time + note_duration 
                        note = pretty_midi.Note(velocity=100, pitch=p + offset, start=start_time, end=end_time)
                        instrument.notes.append(note)
                        
                if all(selected_pitches == i for i in previous_notes):
                    selected_pitches = np.array(selected_pitches)
                    selected_pitches += offset
                    selected_pitches = list(selected_pitches)
                previous_notes.append(selected_pitches)
                
                if len(previous_notes) > max_history:
                    previous_notes = previous_notes[-max_history:]
                start_time = end_time

            pm.instruments.append(instrument)
            pm.write(f"music/1d_VAE_{i}.mid") 
```

```python
# Generate and save new MIDI with note durations
generate_music_with_repetition_control(model, latent_dim, num_samples=3, seq_length=400)
print("Done")
```

### Results
While our generated music wont sound like any of our training data, our baseline would have to be a sample of our training data. To evaluate what a "good" output is, we would need to perform a Subjective Listening Test, either using a mean opinion score. Listeners would have to rate the quality of the composition in comparison to the training data, or a related composition.

# Task 1b - 2D Variational Autoencoder
### Introduction
One major disadvantage of the previous model is its dimensionality. Attempting to generate 400 notes results in an input size of 102,400 Nodes! This can be mitigated by using a 2-D VAE. Since our piano rolls are 2-dimensional (where rows are pitches and columns are time intervals), we can instead opt to create an encoder with <code>Conv2d</code>. Our node size decreases by half, and our runtimes are improved. For our first model, an epoch would take roughly 3 minutes to complete. Our new model takes roughly 37 seconds per epoch.

### Parameters, Arguments, and Variables
The following parameters and their selected arguments have changed compared to the first model:

| Parameter | Description | Argument |
| --------- | ----------- | -------- |
| `input_dim` | Deprecated, is now `self._to_linear` | 256 * 8 * 25 |

This model also chaged the following variable values:

| Variable | Description | Value |
| -------- | ----------- | ----- |
| `num_epochs` | Number of epochs in the loss function | 35 `or` early stopping |
| `threshold` | Probability threshold to qualify for candidacy | 0.04 |

### Code
```python
# Steps 1, 2, and 4 are unmodified, and are not reproduced here for simplicity

# 3. Define 2D Convolutional Encoder with BatchNorm and Dropout
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, dropout_prob=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3), stride=2, padding=1),  # Output: (32, 64, 200)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1), # (64, 32, 100)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1), # (128, 16, 50)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1), # (256, 8, 25)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Compute the flattened size after convolutions
        self._to_linear = 256 * 8 * 25  # based on above dimensions

        self.fc_mu = nn.Linear(self._to_linear, latent_dim)
        self.fc_logvar = nn.Linear(self._to_linear, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_layers(x)  # shape: (batch, 256, 8, 25)
        x = x.view(batch_size, -1)  # flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_length=400):
        super().__init__()
        self.seq_length = seq_length
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.output_dim = output_dim
        # Split into two heads: one for pitch, one for duration
        self.fc_pitch = nn.Linear(hidden_dim, 128 * seq_length)  # pitch output
        self.fc_duration = nn.Linear(hidden_dim, 128 * seq_length)  # duration output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        pitch_logits = self.fc_pitch(h)
        duration_logits = self.fc_duration(h)
        # reshape to (batch, channels=2, pitch=128, time=seq_length)
        pitch_logits = pitch_logits.view(-1, 128, self.seq_length)
        duration_logits = duration_logits.view(-1, 128, self.seq_length)
        # Sigmoid for pitch (binary presence)
        pitch_probs = self.sigmoid(pitch_logits)
        return pitch_probs, duration_logits

class MusicVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.encoder = ConvEncoder(latent_dim)
        # Keep your decoder as is or modify similarly
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim=2*128*seq_length, seq_length=seq_length)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        pitch_probs, duration_logits = self.decoder(z)
        return pitch_probs, duration_logits, mu, logvar

# 5. Setup training
midi_files = (glob("mid/0/0/*.mid") +
              glob("mid/0/1/*.mid") + 
              glob("mid/0/2/*.mid") + 
              glob("mid/0/3/*.mid") + 
              glob("mid/0/4/*.mid"))
dataset = MidiDataset(midi_files, seq_length=400)
dataloader = DataLoader(dataset, batch_size=250, shuffle=True, num_workers=0)

input_dim = 2 * 128 * 400  # 2 channels: piano roll + duration
hidden_dim = 2048
latent_dim = 128

model = MusicVAE(hidden_dim, latent_dim, seq_length=400)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Start Training")
t1 = time.time()
best_loss = float('inf')
loss_increased = 0

# To store the best model
best_model_state = None

num_epochs = 35
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        pitch_probs, duration_logits, mu, logvar = model(batch)
        loss = loss_function(pitch_probs, duration_logits, batch.view(batch.size(0), 2, 128, -1), mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_state = model.state_dict()
    else:
        loss_increased = 1
    if loss_increased:
        print(f"Early stopping triggered after epoch {epoch+1}")
        break

t2 = time.time()
print(f"Finished Training in {round(t2 - t1, 2)}s")

# Load the best model after training
if best_model_state is not None:
    model.load_state_dict(best_model_state)
```

### Music Generation
```python
# 7. Generate new music with note durations
def generate_music_with_repetition_control(model, latent_dim, num_samples=1, seq_length=400, max_notes_per_time=3):
    threshold= 0.04
    max_history = 4  # number of previous notes to compare
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        pitch_probs, duration_logits = model.decoder(z)
        pitch_probs = pitch_probs.cpu().numpy()
        duration_logits = duration_logits.cpu().numpy()

        for i in range(num_samples):
            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            dur_pred = duration_logits[i]
            dur_pred = np.clip(dur_pred, 0, 1)

            previous_notes = [[60]]

            start_time = 0
            for t in range(seq_length):
                # Retrieve pitch probabilities at time t
                pitch_probs_t = pitch_probs[i, :, t]
                total_prob = np.sum(pitch_probs_t)
                pitch_probs_t = pitch_probs_t / total_prob
                # Candidate pitches above threshold
                candidate_pitches = np.where(pitch_probs_t > threshold)[0].tolist()
                candidate_probs = pitch_probs_t[candidate_pitches]
                # If no candidate pitches, shift last pitch
                if len(candidate_pitches) == 0:
                    last_pitch = previous_notes[-1][-1]
                    shifted_pitch = pitch_shift_to_c_major(last_pitch)
                    selected_pitches = [shifted_pitch]
                elif len(candidate_pitches) >= max_notes_per_time:
                    # Select top 'max_notes_per_time' pitches based on probability
                    # Get indices of top probabilities
                    top_indices = np.argsort(candidate_probs)[-max_notes_per_time:][::-1]
                    selected_pitches = [candidate_pitches[idx] for idx in top_indices]
                else:
                    # Random choice among candidates
                    choose_idx = np.random.choice(len(candidate_pitches))
                    choose_pitch = candidate_pitches[choose_idx]
                    # shifted_pitch = pitch_shift_to_c_major(choose_pitch)
                    selected_pitches = [choose_pitch]

                offset = 0
                # Check for repetition
                if all(selected_pitches == i for i in previous_notes):
                    offset = 2

                # Append notes to MIDI
                end_time = start_time
                for p in selected_pitches:
                    note_duration = dur_pred[p, t]
                    if note_duration == 0:
                        note_duration = 0.25
                    end_time = start_time + note_duration
                    note = pretty_midi.Note(velocity=100, pitch=p + offset, start=start_time, end=end_time)
                    instrument.notes.append(note)

                # Prepare for next iteration
                if all(selected_pitches == i for i in previous_notes):
                    selected_pitches = np.array(selected_pitches) + offset
                    selected_pitches = list(selected_pitches)

                previous_notes.append(selected_pitches)
                if len(previous_notes) > max_history:
                    previous_notes = previous_notes[-max_history:]
                start_time = end_time

            pm.instruments.append(instrument)
            pm.write(f"music/2d_VAE_{i}.mid")

# Usage (assuming model and functions are defined)
generate_music_with_repetition_control(model, latent_dim, num_samples=3, seq_length=400)
print("Done")
```

### Results
To improve compositional clarity, sampled single notes are evaluated in the C-major scale.

While the loss function results in a greater value than the previous model, the MIDI files generated sound clearer and more structurally sound. This can be thought of as reducing overfitting on the training dataset to better generalize to unseen music.

# Task 2 - Markov Chains
### Introduction
Our second model continues our use of MIDI files, this time for symbolic conditioned generation. This time, the model aims to generate music given a music scale, as well as a keyword to determine a chord progression to be used. The goal of this model will be to generate music that harmonizes; one that generates notes and chords following a scale pattern. For example, the model can generate a "vi IV I V F Major Hopeful" composition. We are also able to train our model using the entirety of the PDMX dataset, as well as another publicly available dataset: Maestro. Maestro contains over 1000 MIDI files that are longer and more robust than those of PDMX. Most importantly, these compositions contain very few chords, giving us more features related to single note generation.

### Data Cleaning
Given that over 250,000 files were used to train our model, caching results was critical to improve consecutive runtimes. 14 different JSON files are used, each taking over half an hour to create, but  only seconds to load in. For this model, MIDITok and symusic were the primary packages used to process the data. No files were left out, as these packages were able to properly extract all the necessary features.

```python
import os
import time

import json
import ast
import random
from glob import glob
from collections import defaultdict

import numpy as np
from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile

random.seed(42)
start = time.time()
```

### Model Description
The algorithm I chose to implement for this model was an expanded Markov Chain, using n-gram probabilities to generate both pitch and duration for each note. The model determines the next note to play based on a random choice, influenced by the probabilities of the trigrams. Simply put: this algorithm predicts the next musical element based on the current state and its transitional probabilities. The goal of a Markov chain is to increase stylistic coherence. This model differs from the VAE used in Task 1a and 1b as there is no loss function. This becomes a problem as Markov chains can follow a pattern that diverges from the original composition. To prevent this, a few constraints are included. First, since most of the music is in 4/4 time scale, we constrain the generations to beats that conform to this time scale. Additionally, notes not belonging to the given input scale are not used. Finally, keywords determine chord progression that plays on every first beat of a scale. All of these features improve the structural coherence of the composition. Advantages of this model are its ability to train on larger datasets with minimal memory usage, while a disadvantage is its simplicity limits how accurate, coherent, or audibly appealing a composition can become.

Similar to our previous task model, our generated music will be evaluated through Subjective Learning tests. Additionally, since this model is simpler and can possibly generate more coherent music by following more basic conditions, we can also determine audio diversity, and generate an Inception Score from our training data, or from an outside composition. During comparison, features such as chord similarity would improve our score. Baselines for this task would include whether the generated composition effectively follows a sequence of notes, or if its distribution of chords is similar to the training set.

**Warning:** Due to GitHub's file size restrictions, <code>beat_extractions.json</code> (466 MB) and <code>note_extractions.json</code> (231 MB) are not pushed to the project page. Additionally, <code>mid</code>(1.79 GB), the folder that contains all 250,000+ MIDI files, is also not pushed. For page simplicity, the code for this model can be found in <code>Assignment2.ipynb</code>

# Analysis and Final Remarks
Based on the audio outputs from these models, 2-dimensional Variational Autoencoders have resulted in the most audibly appealing MIDI compositions. This model is also the fastest to generate music (given that caching is not used), making it the most effective model this project has developed. While the model has only been trained to generate symbolic musical compositions, VAEs are great for use in continuous audio generation, such as .wav formats. The implementation is similar in style to our models: take as input a melSpectrogram or audio file, encode into a set of parameters, decode back to a melSpectrogram and back to an audio format of choice. Perhaps in the near future this project will be updated to include samples of generated music in this manner.

Additional neural networks and deep learning models to try for symbolic music generation include Generative Adversarial Networks (GANs) and Long Short-Term Memory (LSTMs).

Finally, we should discuss how our datasets have been used in the past. With the increased calls for artist protection against AI copyright infringement, there needed to be useful copyright-free training data to train models while being up to date with modern music trends. 

PDMX has been used to train music generation models and for MIDI conversion applications. While VAEs and Markov Chains are useful for symbolic generation, others have opted to use LSTMs, CNNs, GANs, and HMMs. Results are difficult to compare from a mathematical perspective, but our music generation differs from others in that either chords or single notes were prevalent in most of the music; a combination of both was rarely ever observed.

MAESTRO is a dataset composed of over 200 hours of piano compositions. This data comes in both MIDI and wav formats. This dataset has been used for many ML algorithms related to remote piano composition judging.

**While this method of generating music is no longer the state of the art, the knowledge acquired from this project has vastly improved my understanding of machine learning concepts, and the implementation of Convolutional Neural Networks shows that I can effectively apply these skills to real-world situations.**
