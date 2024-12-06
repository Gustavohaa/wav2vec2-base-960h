
# wav2vec 2.0

O **wav2vec 2.0** aprende representações de fala em dados não rotulados, conforme descrito no artigo [wav2vec 2.0: Um Framework para Aprendizado Autossupervisionado de Representações de Fala (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477).

Também aprendemos representações de fala em vários idiomas no artigo [Aprendizado de Representação Translinguística Não Supervisionada para Reconhecimento de Fala (Conneau et al., 2020)](https://arxiv.org/abs/2006.13979).

Além disso, combinamos o wav2vec 2.0 com auto-treinamento no artigo [Auto-treinamento e Pré-treinamento são Complementares para Reconhecimento de Fala (Xu et al., 2020)](https://arxiv.org/abs/2010.11430).

Para lidar com dados de fala de múltiplos domínios, trabalhamos no artigo [wav2vec 2.0 Robusto: Analisando a Mudança de Domínio no Pré-treinamento Autossupervisionado (Hsu, et al., 2021)](https://arxiv.org/abs/2104.01027).

Finetunamos o **XLSR-53** em múltiplos idiomas para transcrever idiomas não vistos no artigo [Reconhecimento de Fonemas Translinguístico Simples e Eficaz de Zero-shot (Xu et al., 2021)](https://arxiv.org/abs/2109.11680).

## Modelos pré-treinados

Modelo | Divisão de Ajuste Fino | Dataset | Modelo
|---|---|---|---
Wav2Vec 2.0 Base | Sem ajuste fino | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt)
Wav2Vec 2.0 Base | 10 minutos | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_10m.pt)
Wav2Vec 2.0 Base | 100 horas | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_100h.pt)
Wav2Vec 2.0 Base | 960 horas | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt)
Wav2Vec 2.0 Grande | Sem ajuste fino | [Librispeech](http://www.openslr.org/12)  | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt)
Wav2Vec 2.0 Grande | 10 minutos | [Librispeech](http://www.openslr.org/12)  | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_10m.pt)
Wav2Vec 2.0 Grande | 100 horas | [Librispeech](http://www.openslr.org/12)  | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_100h.pt)
Wav2Vec 2.0 Grande | 960 horas | [Librispeech](http://www.openslr.org/12)  | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_960h.pt)

## Treinando um novo modelo com ferramentas de CLI

Dado um diretório contendo arquivos WAV a serem usados para pré-treinamento (recomendamos dividir cada arquivo em segmentos de 10 a 30 segundos).

### Preparar o manifesto de dados de treinamento

Primeiro, instale a biblioteca `soundfile`:

```shell
pip install soundfile
```

Em seguida, execute:

```shell
python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
```

O valor de `$ext` deve ser definido como `flac`, `wav` ou qualquer formato que o `soundfile` possa ler.

O valor de `$valid` deve ser definido como uma porcentagem razoável (como 0.01) dos dados de treinamento a serem usados para validação. Para usar um conjunto de validação pré-definido (como `dev-other` do Librispeech), defina como 0 e substitua `valid.tsv` por um arquivo de manifesto processado separadamente.

### Treinar um modelo base wav2vec 2.0

Essa configuração foi usada para o modelo base treinado no dataset Librispeech no artigo do wav2vec 2.0.

Observação: a entrada esperada deve ser de um canal único, amostrada a 16 kHz.

```shell
$ fairseq-hydra-train     task.data=/path/to/data     --config-dir /path/to/fairseq-py/examples/wav2vec/config/pretraining     --config-name wav2vec2_base_librispeech
```
