import io
import librosa
import matplotlib.pyplot as plt
import numpy as np

# import os
# import shutil

import soundfile as sf
import streamlit as st

# import tempfile

st.set_page_config(layout="wide")
st.title("Musical Analysis")
music_file_extensions = [
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".wma",
    ".ogg",
    ".m4a",
    ".aiff",
    ".alac",
    ".opus",
    ".mid",
    ".midi",
    ".amr",
    ".dsd",
    ".ape",
    ".mka",
    ".webm",
    ".3gp",
    ".aa",
    ".atrac",
    ".au",
    ".ra",
    ".rm",
    ".snd",
    ".tta",
    ".vqf",
    ".wv",
    ".xm",
    ".mod",
    ".s3m",
    ".it",
    ".mtm",
    ".umx",
    ".mo3",
    ".psf",
    ".spc",
    ".nsf",
    ".gbs",
    ".hes",
    ".vgm",
    ".sid",
    ".ay",
    ".ym",
    ".gym",
    ".usf",
    ".ac3",
    ".dts",
    ".ec3",
    ".mlp",
    ".mpc",
    ".ofr",
    ".tak",
    ".tta",
    ".wv",
    ".wma",
    ".w64",
    ".kar",
    ".cda",
    ".m3u",
    ".pls",
    ".asx",
    ".xspf",
]
# col_left, col_right = st.columns(2)


# Function to display file uploader and visualize music
def visualize_music(file):
    st.audio(file)
    fig, ax = plt.subplots(nrows=3, sharex=True)

    y_1, sr_1 = librosa.load(io.BytesIO(file.getvalue()))
    librosa.display.waveshow(y_1, sr=sr_1, ax=ax[0])
    ax[0].set_title("Envelope view, mono")
    ax[0].label_outer()

    y_1s, sr_1s = librosa.load(io.BytesIO(file.getvalue()), mono=False)
    librosa.display.waveshow(y_1s, sr=sr_1s, ax=ax[1])
    ax[1].set_title("Envelope view, Stereo")
    ax[1].label_outer()

    y_harm_1, y_perc_1 = librosa.effects.hpss(y_1)
    librosa.display.waveshow(y_harm_1, sr=sr_1, alpha=0.5, ax=ax[2], label="Harmonic")
    librosa.display.waveshow(
        y_perc_1, sr=sr_1, color="r", alpha=0.5, ax=ax[2], label="Percussive"
    )
    ax[2].set(title="Multiple waveforms")
    ax[2].legend()
    st.pyplot(fig)

    fig_spec, ax_spec = plt.subplots()
    D_f_1 = librosa.amplitude_to_db(np.abs(librosa.stft(y_1)))
    img = librosa.display.specshow(
        D_f_1, y_axis="linear", x_axis="time", sr=sr_1, ax=ax_spec
    )
    ax_spec.set(title="Linear-frequency power spectrogram")
    st.pyplot(fig_spec)


# Streamlit code
st.header("File Music 1")

file_music_1 = st.file_uploader(
    "# File Music 1",
    type=music_file_extensions,
    # label_visibility="hidden",
)

if file_music_1 is not None:
    visualize_music(file_music_1)
