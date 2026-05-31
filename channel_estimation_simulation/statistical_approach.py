import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

FIR_LENGTH = 30
CP_PREFIX = 32
OFDM_LENGTH = 1024 + CP_PREFIX
M = 10
NOISE_VAR = 5

# Generate modulated OFDM symbols 
X_freq = ((2*np.random.randint(0,2,(M,1024))-1) + 1j*(2*np.random.randint(0,2,(M,1024))-1))

# Time domain
x_time = np.fft.ifft(X_freq, axis=1)

# Random FIR channel
g_time = np.random.uniform(-0.4,0.4, FIR_LENGTH)

# Add cyclic prefix
prefix = x_time[:, -CP_PREFIX:]
x_cp = np.concatenate([prefix, x_time], axis=1)

# Pass through channel
y = np.zeros(shape = (x_cp.shape[0],x_cp.shape[1]+FIR_LENGTH-1))
for m in range(M):
    y[m] = np.convolve(x_cp[m], g_time, mode='full')

# Remove CP
y_no_cp = y[:, CP_PREFIX:CP_PREFIX+1024]

# FFT to frequency domain
y_freq = np.fft.fft(y_no_cp, axis=1)

# Add noise complex normal and white
noise = np.random.normal(0, np.sqrt(NOISE_VAR/2), y_freq.shape) + 1j*np.random.normal(0, np.sqrt(NOISE_VAR/2), y_freq.shape)
y_freq_noisy = y_freq + noise

# MLE channel estimation in frequency domain
g_est_freq = np.sum(np.conj(X_freq) * y_freq_noisy, axis=0) / np.sum(np.abs(X_freq)**2, axis=0)

# Back to time domain
g_est_time_full = np.fft.ifft(g_est_freq)
g_est_time = g_est_time_full[:FIR_LENGTH]

# True channel frequency response
H_true_freq = np.fft.fft(np.pad(g_time, (0, 1024-FIR_LENGTH)))

fig, axs = plt.subplots(2, 2, figsize=(8, 6))

# 1. Time domain channel
axs[0, 0].plot(g_time, label='True Channel (Real)')
axs[0, 0].plot(np.real(g_est_time), '--', label='Estimated Channel (Real)')
axs[0, 0].plot(np.imag(g_est_time), ':', label='Estimated Channel (Imag)')
axs[0, 0].set_title('Channel Estimation in Time Domain')
axs[0, 0].set_xlabel('Tap Index')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. Time domain estimation error
axs[0, 1].plot(np.abs(g_time - np.real(g_est_time)), label='Estimation Error (Magnitude)')
axs[0, 1].set_title('Time Domain Estimation Error per Tap')
axs[0, 1].set_xlabel('Tap Index')
axs[0, 1].set_ylabel('Absolute Error')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3. Frequency domain magnitude
axs[1, 0].plot(np.abs(H_true_freq), label='True Channel (Freq Mag)')
axs[1, 0].plot(np.abs(g_est_freq), '--', label='Estimated Channel (Freq Mag)')
axs[1, 0].set_title('Channel Estimation in Frequency Domain (Magnitude)')
axs[1, 0].set_xlabel('Subcarrier Index')
axs[1, 0].set_ylabel('Magnitude')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. Frequency domain phase and estimation error
phase_true = np.angle(H_true_freq)
phase_est = np.angle(g_est_freq)


axs[1, 1].plot(phase_true, label='True Channel Phase')
axs[1, 1].plot(phase_est, '--', label='Estimated Phase')
axs[1, 1].set_title('Frequency Domain Phase & Estimation Error')
axs[1, 1].set_xlabel('Subcarrier Index')
axs[1, 1].set_ylabel('Radians / Error')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
