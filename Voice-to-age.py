import NeuralNet
import RTVoiceData

n_mel = 64

network = NeuralNet.Audio_Transformer(n_mels=64)
voice_data = RTVoiceData.RTVoice(n_mel=64)

while True:
    data = voice_data.collect_voice_data()
    print(network(data))