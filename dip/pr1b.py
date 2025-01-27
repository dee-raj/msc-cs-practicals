from sympy import fft

seq = [15, 21, 13, 14]
decimal_point = 4
transform = fft(seq, decimal_point)

print("Input Sequence:", seq)
print("FFT:", transform)
