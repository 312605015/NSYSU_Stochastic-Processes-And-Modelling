from moviepy.editor import AudioFileClip
from scipy import signal
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Load the audio signals
def load_audio(file_path):
    clip = AudioFileClip(file_path)
    audio = clip.to_soundarray()[:, 0]  # Extract mono channel
    clip.close()
    return audio

# Specify the file paths of the three audio signals (MP4 files)
file_path1 = 'signal1.mp4'
file_path2 = 'signal2.mp4'
file_path3 = 'signal3.mp4'

# Load the audio signals
signal1 = load_audio(file_path1)
signal2 = load_audio(file_path2)
signal3 = load_audio(file_path3)

# Resample the signals to the same sample rate
target_sample_rate = 44100  # Set your desired sample rate
signal1 = signal.resample(signal1, int(signal1.shape[0] * target_sample_rate / signal1.shape[0]))
signal2 = signal.resample(signal2, int(signal2.shape[0] * target_sample_rate / signal2.shape[0]))
signal3 = signal.resample(signal3, int(signal3.shape[0] * target_sample_rate / signal3.shape[0]))

# Ensure all signals have the same length
min_length = min(len(signal1), len(signal2), len(signal3))
signal1 = signal1[:min_length]
signal2 = signal2[:min_length]
signal3 = signal3[:min_length]

# Perform signal mixing
mixing_matrix = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
mixed_signals = np.dot(np.array([signal1, signal2, signal3]).T, mixing_matrix)

# Perform Blind Source Separation using FastICA
ica = FastICA(n_components=3, max_iter=1000, tol=1e-4)  # Increase max_iter and decrease tol for better convergence
recovered_signals = ica.fit_transform(mixed_signals)

# Plot original, mixed, and recovered signals
plt.figure()

# Original signals
plt.subplot(4, 1, 1)
plt.title("Signal 1")
plt.plot(signal1)

plt.subplot(4, 1, 2)
plt.title("Signal 2")
plt.plot(signal2)

plt.subplot(4, 1, 3)
plt.title("Signal 3")
plt.plot(signal3)

# Mixed signals
plt.subplot(4, 1, 4)
plt.title("Mixed Signals")
plt.plot(mixed_signals)

plt.tight_layout()
plt.show()

# Plot recovered signals
plt.figure()

plt.subplot(3, 1, 1)
plt.title("Recovered Signal 1")
plt.plot(recovered_signals[:, 0])

plt.subplot(3, 1, 2)
plt.title("Recovered Signal 2")
plt.plot(recovered_signals[:, 1])

plt.subplot(3, 1, 3)
plt.title("Recovered Signal 3")
plt.plot(recovered_signals[:, 2])

plt.tight_layout()
plt.show()
