with open ("train.txt", "r") as f:
    data = f.read().split()

vocab = set(data)

vocab_string = ""

for word in vocab:
    vocab_string += word + "\n"

with open ("vocab.txt", "w") as f:
    f.write(vocab_string)

