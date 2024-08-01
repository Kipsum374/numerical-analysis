Godfather: import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the model function
def model(x, a, b, c):
    return a * x**2 + b * x + c

# Sample data
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([1, 4, 9, 16, 25])

# Fit the model to the data
params, covariance = curve_fit(model, x_data, y_data)

# Plot the data and the fit
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, model(x_data, *params), label='Fitted curve', color='red')
plt.legend()
plt.show()

print(f"Fitted parameters: {params}")
 Godfather: import sympy as sp

# Define the symbol
x = sp.symbols('x')

# Define the function
f = x**2 - x - 2

# Differentiate the function
f_prime = sp.diff(f, x)

print(f"The derivative of {f} is {f_prime}")
[30/07, 11:47] Godfather: import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

# Perform spline interpolation
spline = splrep(x, y)
x_new = np.linspace(1, 5, 100)
y_new = splev(x_new, spline)

# Plot the data and the spline interpolation
plt.scatter(x, y, label='Data')
plt.plot(x_new, y_new, label='Spline interpolation', color='red')
plt.legend()
plt.show()
[30/07, 11:47] Godfather: import scipy.integrate as spi
import numpy as np

# Define the function
def f(x):
    return x**2 - x - 2

# Integrate the function from 1 to 3
integral, error = spi.quad(f, 1, 3)

print(f"The integral of the function from 1 to 3 is {integral} with an error of {error}")
[30/07, 11:47] Godfather: import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
f1 = 50 # Frequency of first sine wave
f2 = 120 # Frequency of second sine wave
fs = 1000 # Sampling frequency
T = 1 # Duration in seconds

# Time vector
t = np.linspace(0, T, fs*T, endpoint=False)

# Signal definition
s = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# Compute FFT
S = np.fft.fft(s)
# Compute frequencies
frequencies = np.fft.fftfreq(len(S), 1/fs)

# Only take the positive frequencies
positive_frequencies = frequencies[:len(frequencies)//2]
positive_S = np.abs(S[:len(S)//2]) # Magnitude of FFT

# Plot the signal
plt.figure(figsize=(12, 6))

# Plot time domain signal
plt.subplot(2, 1, 1)
plt.plot(t, s)
plt.title('Time Domain Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot frequency domain signal
plt.subplot(2, 1, 2)
plt.plot(positive_frequencies, positive_S)
plt.title('Frequency Domain Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()
plt.tight_layout()
plt.show()
 Godfather: import numpy as np

def gradient_descent(grad_f, initial_guess, learning_rate, num_iterations):
    x, y = initial_guess
    for _ in range(num_iterations):
        grad_x, grad_y = grad_f(x, y)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
    return x, y

def grad_f(x, y):
    df_dx = 2*x - y + 1
    df_dy = 2*y - x - 1
    return np.array([df_dx, df_dy])

# Initial guess
initial_guess = (0, 0)
# Learning rate
learning_rate = 0.1
# Number of iterations
num_iterations = 100

# Perform gradient descent
min_x, min_y = gradient_descent(grad_f, initial_guess, learning_rate, num_iterations)
print(f"Minimum value found at x = {min_x}, y = {min_y}")
[30/07, 11:47] Godfather: import numpy as np
import matplotlib.pyplot as plt

# Define the data points
x_points = np.array([1, 2, 3, 4])
y_points = np.array([1, 4, 9, 16])

def lagrange_interpolation(x_points, y_points, x):
    """
    Perform Lagrange interpolation for the given data points.
    
    Parameters:
    x_points : array-like
        The x coordinates of the data points.
    y_points : array-like
        The y coordinates of the data points.
    x : float
        The x value at which to evaluate the interpolating polynomial.
    
    Returns:
    float
        The interpolated value at x.
    """
    n = len(x_points)
    total = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term = term * (x - x_points[j]) / (x_points[i] - x_points[j])
        total += term
    return total

# Define the x values for plotting the polynomial
x_values = np.linspace(1, 4, 100)
y_values = [lagrange_interpolation(x_points, y_points, x) for x in x_values]

# Plot the data points and the interpolating polynomial
plt.plot(x_points, y_points, 'o', label='Data points')
plt.plot(x_values, y_values, label='Lagrange polynomial')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lagrange Polynomial Interpolation')
plt.legend()
plt.grid(True)
plt.show()