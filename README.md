# Kokoro TTS Web Interface

A powerful text-to-speech web application using the Kokoro TTS model for generating high-quality, natural-sounding speech from text.

## Features

- **High-Quality Speech Synthesis**: Utilizes the Kokoro TTS model for natural-sounding speech generation
- **Multiple Voices**: Supports 30+ different voices in American and British English (male and female)
- **Extended Long Text Support**: Processes text of any length by intelligently breaking it into optimal chunks, extending beyond the base model's limitations
- **Adaptive Pauses**: Dynamically adjusts pauses between sentences based on content for more natural speech flow
- **Caching System**: Stores processed audio segments to speed up repeat processing
- **Speed Control**: Adjust speech speed from 0.5x to 2.0x
- **User-Friendly Interface**: Simple Gradio web UI for easy interaction

## Installation

1. Clone this repository:
```bash
git clone https://github.com/tejasundeep/text-to-speech.git
cd tts
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the provided URL (typically http://127.0.0.1:7860)

3. Enter your text, select a voice, adjust the speed if desired, and click "Generate Speech"

4. The generated audio will appear in the output section, where you can play or download it

## Advanced Options

- **Cache Management**: Enable/disable caching or clear the cache in the Advanced Options panel
- **Adaptive Pauses**: Toggle dynamic pause adjustment between sentences
- **Processing Large Texts**: The system automatically handles large texts by breaking them into manageable chunks

## Key Improvements Over Base Model

- **Long Text Processing**: While the base Kokoro model has limitations with long texts, this application implements intelligent chunking algorithms that automatically process text of any length
- **Error Handling**: Robust retry mechanisms and fallback processing for challenging text segments
- **Performance Optimization**: Caching system dramatically improves performance for repeated text
- **Natural Speech Flow**: Custom pause calculations between chunks create more natural-sounding output

## Technical Details

The application uses:
- Kokoro TTS model (82M parameter version) from Hugging Face
- Gradio for the web interface
- Advanced text preprocessing for improved pronunciation
- Intelligent chunking algorithms for handling long texts
- Caching system for optimized performance
- Adaptive pause calculation based on content and voice characteristics

## Requirements

- Python 3.7+
- PyTorch
- Gradio
- Kokoro TTS library
- SoundFile
- NumPy

## Project Structure

- `app.py`: Main application file containing the TTS logic and Gradio interface
- `output_audio/`: Directory where generated audio files are stored
- `tts_cache/`: Directory for the caching system

## License

MIT License

Copyright (c) 2023 Teja Sundeep

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- [Kokoro TTS model](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad
- This project extends the original model capabilities with improved text processing 
