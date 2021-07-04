from numpy import random
import torch
import torch.nn as nn

# Wczytujemy książkę
with open('Wiedzmin.txt', 'r', encoding="latin-1") as f:
    lines = f.readlines()
    lines = lines[1:]
    chars = ''.join(lines)


# Map między znakami a liczbami
# Wzięte z https://github.com/pytorch/examples/blob/master/word_language_model/data.py
class Dictionary(object):
    def __init__(self):
        self.char2indx = {}
        self.indx2char = []

    def add_char(self, char):
        if char not in self.char2indx:
            self.indx2char.append(char)
            self.char2indx[char] = len(self.indx2char) - 1
        return self.char2indx[char]

    def __len__(self):
        return len(self.indx2char)


# Konwertujemy dane z znaków na tokeny
data_dictionary = Dictionary()
tensor_data = torch.LongTensor(len(chars))

for x, y in enumerate(chars):
    tensor_data[x] = data_dictionary.add_char(y)
n_elements = len(data_dictionary)

# definiujemy nasz Model
# Posługiwałem się oficjalną biblioteką PyTorch https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# oraz https://discuss.pytorch.org/t/lstm-init-hidden-to-gpu/81441
class Model(nn.Module):
    def __init__(self, input_size, batch_size, rnn_module="RNN", hidden_size=64, num_layers=1, dropout=0):
        super(Model, self).__init__()
        self.input_size = input_size
        self.rnn_module = rnn_module
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if rnn_module == "RNN":
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        elif rnn_module == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        elif rnn_module == "GRU":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden):
        output = input.view(1, -1, self.input_size)
        output, hidden = self.rnn(output, hidden)
        output = self.output(output[0])
        return output, hidden

    def initHidden(self, batch_size):
        # initialize hidden state to zeros
        if self.rnn_module == "LSTM":
            return torch.zeros(self.num_layers, batch_size, self.hidden_size), torch.zeros(
                self.num_layers, batch_size, self.hidden_size)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# ładujemy nasz model do PATH_MODEL
PATH_MODEL = 'wandb/run-20210423_145031-14qno0sn/files/model.pt'
model = torch.load(PATH_MODEL)
model = model.to(torch.device("cpu"))

with torch.no_grad():

    model.eval()

    # Вуаштшгоуьн sekwencję znaków aby zainicjować stany ukryte
    init_chars = "Yennefer "

    init_data = torch.LongTensor(len(init_chars))
    for i, c in enumerate(init_chars):
        init_data[i] = data_dictionary.char2indx[c]

    # Przetwarzamy w one-hot
    init_data = torch.zeros(len(init_data), len(data_dictionary)).scatter_(1, init_data.unsqueeze(-1), 1)

    # Inicjujemy ukrytą warstwę i przesyłamy sekwencję znaków do modelu
    hidden = model.initHidden(1)
    for init_char in init_data:
        output, hidden = model(init_char, hidden)

    # Przewidujemy kolejne znaki pojedynczo
    numberOfChars = 1000
    chars = init_chars
    for _ in range(numberOfChars):
        # Obliczamy rozkład prawdopodobieństwa wyjść o temperaturze 0,5
        prob = nn.Softmax(1)(output / 0.5).squeeze().numpy()

        # Próbka z wyjść
        output_idx = random.choice(len(prob), p=prob)

        # Zapisujemy przewidywany znak
        predicted_char = data_dictionary.indx2char[output_idx]
        chars += predicted_char

        # Przekształcamy przewidywany znak w one-hot wektor
        output_idx = torch.LongTensor([[output_idx]])
        next_input = torch.zeros(len(output_idx), len(data_dictionary)).scatter_(1, output_idx, 1)

        # Zapisujemy do Modelu aby przewidzieć następny znak
        output, hidden = model(next_input, hidden)

    # drukujemy przewidywaną sekwencję
    print("Początek tekstu: ", init_chars)
    print("Przewidywana sekwencja:\n", chars)

    #Zapisujemy nasz wygenerowany tekst do pliku
    textGen = open("Wiedzmit - Generated Text.txt","w",encoding="latin1")
    textGen.write(init_chars)
    textGen.write(chars)
    textGen.close()