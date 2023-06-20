# Speaker Clustering

With the code of this repository it is possible to create clusters from a set of audio files. For this, speaker embeddings extracted using the Speakernet or Titanet models, present in NVIDIA's NeMO repository, are used. Just provide a folder containing the audio files that, executing this code, will result in the division of the audios into clusters according to the speaker embeddings.

## Requirements

To extract the embeddings using the SpeakerNet or TitaNet models, it is necessary to install the following requirements:

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython tqdm
pip install nemo_toolkit['all']
```
It is also necessary to install the Pytorch framework:

```bash
$ pip install torch torchaudio
```

Alternatively, you can create a conda env using the available yml file:

```bash
$ conda env create -f environment.yml
```

Then just activate the virtual environment:

```bash
conda activate speaker_clustering
```

## Dataset

The dataset must be in the following format:

```sh
- input
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
python main.py --input=input --output=output
```


The following steps will be performed:

```sh
Audios Normalizing >> Embeddings Extraction >> Clustering >> Copying Files >> Plotting
```