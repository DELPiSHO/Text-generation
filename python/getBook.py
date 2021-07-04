with open('Wiedzmin - Copy.txt', 'r', encoding="latin-1") as f:
    lines = f.readlines()
    chars = ''.join(lines)

# Test code
print("Wiedźmin")
print("Total number of chars: ", len(chars))
print("Unique chars: ", len(set(chars)))