from scipy.fft import fft, fftfreq
import numpy as np

def get_main_fourier_component(t, y, ignore_dc=True):
	'''take the fourier transform of a timesignal and return the dominant fourier component'''

	N = len(t)
	yf = fft(y)
	if ignore_dc:
		yf[0] = 0 # remove DC component from FFT
	dt = t[1]-t[0] # time step size, assume equal
	xf = fftfreq(N, dt)[:N//2] # construct xaxis

	return xf[np.argmax(np.abs(yf))]