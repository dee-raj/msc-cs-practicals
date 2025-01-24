from sympy import fft

# Define the input sequence
seq = [15, 21, 13, 14]

# Number of decimal points to display in the FFT output
decimal_point = 4

# Compute the FFT of the sequence
transform = fft(seq, decimal_point)

# Display the result
print("Input Sequence:", seq)
print("FFT:", transform)
