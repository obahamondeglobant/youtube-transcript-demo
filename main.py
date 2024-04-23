import re
import tempfile
from typing import Generator

import numpy as np
import pydub
import pytube
import streamlit as st
import torch
from requests import get
from whisper import load_model, transcribe

SAMPLE_RATE = 16000
CHUNK_DURATION = 3
CHUNKSIZE = SAMPLE_RATE * CHUNK_DURATION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("tiny").to(DEVICE)


def transcribe_youtube(url: str) -> Generator[str, None, None]:
    yt = pytube.YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    with tempfile.TemporaryDirectory() as tmpdir:
        stream.download(output_path=tmpdir)
        audio_file_path = f"{tmpdir}/{stream.default_filename}"
        audio = pydub.AudioSegment.from_file(audio_file_path)
        audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
        audio_samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (
            2**15
        )
        for i in range(0, len(audio_samples), CHUNKSIZE):
            chunk = audio_samples[i : i + CHUNKSIZE]
            if len(chunk) < CHUNKSIZE:
                chunk = np.pad(chunk, (0, CHUNKSIZE - len(chunk)), mode="constant")
            text = transcribe(model, chunk)
            yield text["text"]


def search_videos(query: str):
    search_url = f"https://www.youtube.com/results?search_query={query}"
    response = get(
        search_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        },
    )
    data = response.text
    pattern = re.compile(r"watch\?v=(\S{11})")
    videos = set[str]()
    for video_id in pattern.findall(data):
        videos.add(f"https://www.youtube.com/watch?v={video_id}")
    return list(videos)


def main():
    st.title("Transcripción de videos de YouTube")
    st.write(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@0.6.4/dist/tailwind.min.css">',
        unsafe_allow_html=True,
    )
    query = st.text_input("Ingrese la consulta:")

    if st.button("Buscar"):
        if query:
            videos = search_videos(query)
            if videos:
                for i, video in enumerate(videos):
                    st.write(
                        f"""<a href='/?v={video[32:43]}' class="block bg-gray-800 p-4 rounded-lg mb-4 decoration-none cursor-pointer">
						<p
						class="text-cyan-300 hover:underline"
						>{pytube.YouTube(video).title}</p>
						<img
						class="rounded-lg w-96  cursor-pointer"
						src='{pytube.YouTube(video).thumbnail_url}' /></a>""".format(
                            video
                        ),
                        unsafe_allow_html=True,
                    )

    if "v" in st.query_params:
        url = f"https://www.youtube.com/watch?v={st.query_params['v']}"
        st.write(f"Transcripción de {pytube.YouTube(url).title}")
        for text in transcribe_youtube(url):
            st.write(text)


if __name__ == "__main__":
    main()
