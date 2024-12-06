# Wav2Vec2-Base-960h
**Facebook AI's Wav2Vec2 Model**

## Descrição
O modelo **Wav2Vec2-Base-960h** é uma versão base do Wav2Vec2, pré-treinado e ajustado utilizando 960 horas do dataset Librispeech, com amostras de áudio em 16 kHz. Para obter resultados ideais, certifique-se de que a entrada de áudio também esteja amostrada a 16 kHz.

---

## Publicação
**Título do Paper:** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*

**Autores:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli  
**Data de Publicação:** 24 de setembro de 2020  

**Resumo:**  
O Wav2Vec 2.0 introduz um modelo que aprende representações poderosas a partir de áudio não anotado e, posteriormente, é ajustado em dados transcritos. Este modelo é capaz de superar os melhores métodos semi-supervisionados, mesmo utilizando menos dados anotados.  

Os experimentos demonstraram resultados impressionantes em benchmarks como o LibriSpeech, com WERs de 1.8/3.3 (clean/other) utilizando todos os dados anotados e até 4.8/8.2 com apenas 10 minutos de dados anotados e 53.000 horas de dados não anotados.

Leia o paper completo [aqui](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20).

---

## Exemplos de Uso
### Transcrição de Áudio
O modelo pode ser utilizado para transcrever arquivos de áudio como segue:

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

# Carregar o modelo e o processador
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Carregar dataset de exemplo
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# Tokenizar o áudio
input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values

# Obter os logits
logits = model(input_values).logits

# Decodificar a transcrição
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print("Transcrição:", transcription)
