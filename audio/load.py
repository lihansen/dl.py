# %% 
import torchaudio

waveform, sr = torchaudio.load("/home/hansen/dl.py/audio/61-70968-0000.flac")

# %%
waveform.shape