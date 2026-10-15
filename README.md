# Vietnamese Traditional Music Genre Classification (CRNN and PCRNN)

Keras notebooks that classify Vietnamese traditional music into 5 genres with two
CNN + RNN models: a **Convolutional Recurrent Neural Network (CRNN)** and a
**Parallel Convolutional Recurrent Neural Network (PCRNN)**.

This is the code behind the ICCRI 2024 paper
*"Exploring CNN-based Architectures for Vietnamese Traditional Music Genre Classification"*.

## The problem

Music genre classification helps music apps browse, search and recommend songs.
Most research in this area uses Western music. Vietnamese traditional music has its
own instruments, singing styles and regional variations, and it has received much
less attention. This project compares two CNN-based architectures that also model
how the sound changes over time, and checks which one works better for
Vietnamese traditional genres.

## Dataset

We use the **VNTM dataset on Kaggle**:
<https://www.kaggle.com/datasets/homata123/vntm-for-building-model-5-genres/data>

It has 30-second `.wav` clips in 5 genres. Clip counts below come from the saved
outputs of `notebooks/dataanalysis.ipynb`:

| Genre | Vietnamese name | Folder | 30 s clips | 10 s segments |
|---|---|---|---:|---:|
| Cai Luong | Cải lương | `cailuong` | 500 | 1,500 |
| Ca Tru | Ca trù | `catru` | 499 | 1,483 |
| Chau Van | Chầu văn | `chauvan` | 500 | 1,498 |
| Cheo | Chèo | `cheo` | 500 | 1,500 |
| Xam | Xẩm | `hatxam` | 500 | 1,500 |
| **Total** | | | **2,499** | **7,481** |

The 10-second segments are made by `notebooks/dataanalysis.ipynb`, which cuts each
clip into 10-second pieces. They are the input for PCRNN.

Both models use a stratified random split: 80% training, 10% validation, 10% test
(`random_state=42`).

## Models

Both models are built and trained in `notebooks/CRNN.ipynb`. Feature extraction
is in `notebooks/utils.py`.

| | CRNN | PCRNN |
|---|---|---|
| Audio | Full clip, resampled to 12 kHz, trimmed or padded to 29.12 s | 10-second segment |
| Input | Log-mel spectrogram, 96 mel bands × 1366 frames × 1 channel (FFT 512, hop 256) | Log-magnitude STFT (FFT 1024, hop 512), cut or padded to 513 frequency bins × 128 frames, fed to two branches: a 513 × 128 × 3 "image" for the CNN branch and a 513 × 128 matrix for the RNN branch |
| Structure | 4 convolution blocks (Conv2D, batch normalization, ELU, max pooling, dropout) followed by 2 GRU layers (32 units) | Two branches in parallel: a CNN branch (5 Conv2D + max pooling blocks with ReLU, batch normalization, dropout) and an RNN branch (1D max pooling + bidirectional GRU, 128 units). Their outputs are joined into one vector |
| Output | Dense layer, 5 units, sigmoid | Dense layer, 5 units, softmax |
| Parameters (saved summary) | 393,893 | 373,477 |
| Training | Adam (lr 0.001), binary cross-entropy, batch 32, up to 100 epochs, early stopping | Adam, binary cross-entropy, batch 32, up to 100 epochs, early stopping |

CRNN follows Choi et al. (2016). PCRNN follows Yang et al. (2020); as described in
the paper, we removed the embedding layer of the original RNN branch. The best
weights (by validation accuracy) are saved as checkpoints during training.

Some training settings in the notebook (for example the CRNN batch size and
early-stopping patience) are not the same as the ones described in the paper.

## Results

### Single run in this notebook

These numbers come from the **saved outputs of `notebooks/CRNN.ipynb`**. Each
model was trained and tested **once**, so they are a single run, not an average.
Values are rounded to 4 decimals.

| Model | Test set | AUC ROC | F1 (mean over classes) | Precision | Accuracy |
|---|---|---:|---:|---:|---:|
| CRNN | 250 clips (30 s) | 0.9868 | 0.9484 | 0.9516 | 0.9480 * |
| PCRNN | 749 segments (10 s) | 0.9961 | 0.9853 | 0.9853 | 0.9853 |

\* **Note on the CRNN accuracy.** The old code printed the precision variable on
the line labeled `Accuracy Score`, so the saved output line labeled "Accuracy"
(0.9516129) actually shows **precision**. The code is now fixed, but the notebook
was not re-run, so the saved output still shows the old line. The 0.9480 in the
table is worked out from the saved output of the next cell, which lists the
correctly predicted test clips: 237 of 250.

The two models use different inputs (30-second clips vs 10-second segments), so
their test sets are not the same.

### Reported in the paper

The paper reports the average over several training and test runs
(Table I, mean ± variation):

| Model | F1-score | Accuracy |
|---|---:|---:|
| CRNN | 95.1% ± 0.6% | 94.6% ± 1.2% |
| PCRNN | 97.4% ± 1.1% | 97.4% ± 1.3% |

In both the paper and this notebook, PCRNN scores higher than CRNN.

## Paper

Nguyen Do Nhat Huy, Le Do Thanh Hung, Mai Anh Quan, Huynh Anh Dung, Tieu Nhat Thanh,
Do Quang Huy, Bui Tung Hung.
**"Exploring CNN-based Architectures for Vietnamese Traditional Music Genre Classification."**
ICCRI 2024. FPT University, Ho Chi Minh City, Vietnam.

- First author: Nguyen Do Nhat Huy. Corresponding author: Mai Anh Quan (repository owner).
- PDF: [`docs/CRI008_Exploring_CNN-based_Architectures_for_Vietnamese_Traditional_Music_Genre_Classification.pdf`](docs/CRI008_Exploring_CNN-based_Architectures_for_Vietnamese_Traditional_Music_Genre_Classification.pdf)
  — **preprint / draft**. It may differ from the final published version.

```bibtex
@inproceedings{huy2024exploring,
  title     = {Exploring CNN-based Architectures for Vietnamese Traditional Music Genre Classification},
  author    = {Nguyen Do Nhat Huy and Le Do Thanh Hung and Mai Anh Quan and Huynh Anh Dung and Tieu Nhat Thanh and Do Quang Huy and Bui Tung Hung},
  booktitle = {ICCRI 2024},
  year      = {2024}
}
```

**Related work by the same team:** a second paper, published at JCSSE 2024 (IEEE),
compares DenseNet-based models on Vietnamese traditional music genre classification.
Code: [github.com/Bui-Tung-Hung/LateFusionCNN-DenseNet121-VNTM-MGC](https://github.com/Bui-Tung-Hung/LateFusionCNN-DenseNet121-VNTM-MGC)

**Award:** the team won **3rd prize in the Research category at FPT ResFes 2024** for this research.

## Team & my role

This was a 2024 research project by a team of 7 at FPT University,
Ho Chi Minh City (the authors listed above).

My role (Mai Anh Quan, [@Qcymaq](https://github.com/Qcymaq)):

- Corresponding author of the ICCRI 2024 paper.
- Wrote the CRNN and PCRNN notebooks in this repository.
- Third author of the related JCSSE 2024 IEEE paper.

The paper thanks Mr. Do Duc Hao (FPT University) for his guidance and
Mr. Nguyen Minh Dang for reviewing the paper.

## How to run

The saved outputs were produced with Python 3.11 on a Windows PC. According to the
paper, the experiments ran on a desktop PC with an NVIDIA GTX 1650 GPU and 16 GB of RAM.

1. **Install ffmpeg.** `pydub` needs it to read and write audio
   (for example `sudo apt install ffmpeg`, `brew install ffmpeg` or `winget install ffmpeg`).
2. **Create an environment and install the packages:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt jupyter
   ```

3. **Download the dataset into `data/`.** The notebooks read all paths from one
   `DATA_DIR` variable near the top (default `"data/"`). The path is relative to
   the notebook, and Jupyter runs notebooks from the `notebooks/` folder, so by
   default the data goes in `notebooks/data/`. Change `DATA_DIR` if you keep it
   somewhere else. The notebooks expect this layout (move or rename the folders
   if your download looks different):

   ```text
   notebooks/data/
   ├── VNTM3/                 # original 30 s clips (download from Kaggle)
   │   ├── cailuong/          # CaiLuong.000.wav, CaiLuong.001.wav, ...
   │   ├── catru/             # Catru.000.wav, ...
   │   ├── chauvan/           # Chauvan.000.wav, ...
   │   ├── cheo/              # Cheo.000.wav, ...
   │   └── hatxam/            # Xam.000.wav, ...
   ├── dataset10seconds/      # created by dataanalysis.ipynb
   └── checkpoint/            # created during training
   ```

   `data/` is in `.gitignore`, so the audio and model weights are not committed.

4. **Start Jupyter** (or open the notebooks in VS Code):

   ```bash
   jupyter notebook notebooks/
   ```

5. **Explore the data and create the 10-second segments:**
   run `notebooks/dataanalysis.ipynb`. The "For data 10 seconds" section writes
   `dataset10seconds/`, which PCRNN needs.
6. **Train and evaluate:** run `notebooks/CRNN.ipynb`. The first section trains
   CRNN, the second trains PCRNN. Checkpoints are saved in
   `checkpoint/lcrnn/` and `checkpoint/pcrnn/` inside `DATA_DIR`.

## Repository structure

```text
notebooks/CRNN.ipynb          CRNN and PCRNN: training and evaluation
notebooks/dataanalysis.ipynb  data exploration, spectrogram plots, 10-second splitting
notebooks/utils.py            audio loading, log-mel and STFT features, audio splitting
docs/                         paper (preprint / draft PDF)
requirements.txt              Python packages used by the notebooks
```

## Contact

Mai Anh Quan, Ho Chi Minh City, Vietnam.
Email: quanmaianh.work@gmail.com · GitHub: [Qcymaq](https://github.com/Qcymaq) ·
LinkedIn: [maq159](https://www.linkedin.com/in/maq159)

## License

See [LICENSE](LICENSE) (MIT License).

Please do not reupload without permission.
