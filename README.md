# Speaker Clustering

With the code of this repository it is possible to create clusters from a set of audio files. For this, speaker embeddings extracted using the [Ecapa-TDNN model](https://arxiv.org/abs/2005.07143), available at [huggingface](https://huggingface.co/yangwang825/ecapa-tdnn-vox2). Just provide a folder containing the audio files that, executing this code, will result in the division of the audios into clusters according to the speaker embeddings.

## Requirements

To extract the embeddings using the SpeakerNet or TitaNet models, it is necessary to install the torch and torchaudio requirements:

```bash
$ pip install torch torchvision torchaudio
```

It is also necessary to install the other requirements:

```bash
$ pip install -r requirements.txt
```


## Dataset

The dataset must be in the following format:

```sh
- input
    - folder_0
        - audio1.wav
        - audio2.wav
        - ...
        - audioN.wav
    - folder_1
        - audio1.wav
        - audio2.wav
        - ...
        - audioN.wav
        ...
    - folder__N
        - audio1.wav
        - audio2.wav
        - audioN.wav
```

The result will be as follows:
```sh
- output
    - speaker_0
        - audio1.wav
        - audio2.wav
        - ...
        - audioN.wav
    - speaker_1
        - audio1.wav
        - audio2.wav
        - ...
        - audioN.wav
        ...
    - speaker_N
        - audio1.wav
        - audio2.wav
        - audioN.wav
```


## Executing

```sh
python main.py --input=input --output=output -e wav
```


The following steps will be performed:

```sh
Audios Normalizing >> Embeddings Extraction >> Clustering >> Copying Files >> Plotting
```