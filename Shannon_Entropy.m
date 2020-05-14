% Shannon Entropy

p = [0.01:0.01:1];
b = [1:-0.01:0.01];

S = - (p.*log2(p) + b.*log2(b))

plot(p, S)

