import gradio as gr
from kokoro import KPipeline
import soundfile as sf
import warnings
import torch.nn.utils.parametrizations as param_utils
import numpy as np
import re
import os
import uuid
import time
import json
import hashlib
from pathlib import Path
import gc
import torch

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

# Create output directory if it doesn't exist
AUDIO_OUTPUT_DIR = Path("output_audio")
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

# Create cache directory for storing processed chunks
CACHE_DIR = Path("tts_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize the Kokoro TTS pipeline with explicit repo_id
pipeline = KPipeline(
    lang_code='a',  # 'a' for American English
    repo_id='hexgrad/Kokoro-82M'
)

# Available voices with friendly labels and emojis
voices = {
    '🇺🇸 🚺 Heart ❤️': 'af_heart',
    '🇺🇸 🚺 Bella 🔥': 'af_bella',
    '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇺🇸 🚹 Puck': 'am_puck',
    '🇺🇸 🚹 Echo': 'am_echo',
    '🇺🇸 🚹 Eric': 'am_eric',
    '🇺🇸 🚹 Liam': 'am_liam',
    '🇺🇸 🚹 Onyx': 'am_onyx',
    '🇺🇸 🚹 Santa': 'am_santa',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 Emma': 'bf_emma',
    '🇬🇧 🚺 Isabella': 'bf_isabella',
    '🇬🇧 🚺 Alice': 'bf_alice',
    '🇬🇧 🚺 Lily': 'bf_lily',
    '🇬🇧 🚹 George': 'bm_george',
    '🇬🇧 🚹 Fable': 'bm_fable',
    '🇬🇧 🚹 Lewis': 'bm_lewis',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
}

# Voice characteristics for better prosody handling
voice_characteristics = {
    'af_heart': {'pause_multiplier': 1.0, 'pitch': 'medium'},
    'af_bella': {'pause_multiplier': 0.9, 'pitch': 'medium-high'},
    'af_nicole': {'pause_multiplier': 0.85, 'pitch': 'medium'},
    'am_michael': {'pause_multiplier': 1.1, 'pitch': 'medium-low'},
    'am_fenrir': {'pause_multiplier': 1.2, 'pitch': 'low'},
    'bf_emma': {'pause_multiplier': 0.9, 'pitch': 'medium-high'},
    'bm_george': {'pause_multiplier': 1.1, 'pitch': 'medium-low'},
}

# Default values for voices not specifically defined
default_voice_char = {'pause_multiplier': 1.0, 'pitch': 'medium'}

def get_voice_characteristics(voice_id):
    """Get the characteristics of a specific voice for prosody adjustment"""
    return voice_characteristics.get(voice_id, default_voice_char)

def get_cache_key(text, voice_id, speed):
    """Generate a unique cache key for a chunk of text with specific voice and speed"""
    # Create a deterministic hash from the input parameters
    content = f"{text}|{voice_id}|{speed}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def check_cache(cache_key):
    """Check if a cached audio file exists for the given key"""
    cache_file = CACHE_DIR / f"{cache_key}.npy"
    if cache_file.exists():
        try:
            audio_data = np.load(cache_file)
            return audio_data
        except Exception:
            return None
    return None

def save_to_cache(cache_key, audio_data):
    """Save processed audio to cache"""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.npy"
        np.save(cache_file, audio_data)
    except Exception as e:
        print(f"Error saving to cache: {e}")

def clean_text(text):
    """
    Clean and prepare text for TTS processing with enhanced preprocessing.
    
    Args:
        text (str): The input text
    
    Returns:
        str: Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize common abbreviations for better pronunciation
    abbreviations = {
        r'\bMr\.': 'Mister',
        r'\bDr\.': 'Doctor',
        r'\bSt\.': 'Street',
        r'\bMrs\.': 'Misses',
        r'\bNo\.': 'Number',
        r'\be\.g\.': 'for example',
        r'\bi\.e\.': 'that is',
        r'\bvs\.': 'versus',
        r'\bapprox\.': 'approximately',
        r'\bdept\.': 'department',
    }
    
    for abbr, expansion in abbreviations.items():
        text = re.sub(abbr, expansion, text)
    
    # Replace problematic characters that might cause issues
    text = text.replace('"', "'")
    text = text.replace('…', '...')
    text = text.replace('&', ' and ')
    
    # Normalize numbers for better speech (e.g., 1,000 -> one thousand)
    # This could be expanded with a more comprehensive number-to-text conversion
    
    # Ensure text ends with punctuation
    if text and not text[-1] in '.!?':
        text += '.'
        
    return text

def detect_sentence_type(sentence):
    """Detect the type of sentence for better prosody handling"""
    sentence = sentence.strip()
    if not sentence:
        return 'neutral'
    
    if sentence.endswith('?'):
        return 'question'
    elif sentence.endswith('!'):
        return 'exclamation'
    elif re.search(r'".*?"', sentence):
        return 'quote'
    else:
        return 'statement'

def handle_long_sentences(text, max_length=30):
    """
    Break extremely long sentences into smaller parts to help processing.
    
    Args:
        text (str): Input text
        max_length (int): Maximum words per sentence before forced break
        
    Returns:
        str: Text with long sentences broken down
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    
    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_length:
            # Break long sentence into parts at logical points
            # Try to break at conjunctions first
            conjunction_pattern = r'(?<= and | but | or | because | since | although )'
            parts = re.split(conjunction_pattern, sentence)
            
            if len(parts) > 1:
                # We found natural break points
                processed_parts = []
                for part in parts:
                    part_words = part.split()
                    if len(part_words) > max_length:
                        # Still too long, break by word count
                        subparts = []
                        for i in range(0, len(part_words), max_length):
                            subpart = ' '.join(part_words[i:i+max_length])
                            if not subpart.endswith(('.', '!', '?', ',', ';', ':')):
                                subpart += ','
                            subparts.append(subpart)
                        processed_parts.append(' '.join(subparts))
                    else:
                        processed_parts.append(part)
                result.append(' '.join(processed_parts))
            else:
                # No conjunctions found, break by commas or pure word count
                comma_parts = re.split(r'(?<=,)\s+', sentence)
                if len(comma_parts) > 1:
                    result.append(' '.join(comma_parts))
                else:
                    # Last resort: break by word count
                    parts = []
                    for i in range(0, len(words), max_length):
                        part = ' '.join(words[i:i+max_length])
                        if not part.endswith(('.', '!', '?', ',', ';', ':')):
                            part += ','
                        parts.append(part)
                    result.append(' '.join(parts))
        else:
            result.append(sentence)
    
    return ' '.join(result)

def split_text_into_chunks(text, max_chunk_size=80):
    """
    Split text into smaller chunks based on punctuation and size limits.
    
    Args:
        text (str): The input text to split
        max_chunk_size (int): Maximum number of words per chunk
        
    Returns:
        list: List of text chunks
    """
    if not text:
        return []
    
    # Handle edge case of extremely long sentences first
    text = handle_long_sentences(text)
    
    # First split by paragraphs
    paragraphs = text.split('\n')
    chunks = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_word_count = len(sentence.split())
            
            # If this single sentence is too long, break it down
            if sentence_word_count > max_chunk_size:
                # If we have an existing chunk, add it first
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                
                # Split long sentence by commas or natural breaks
                sentence_parts = re.split(r'(?<=,|\-)\s+', sentence)
                current_part = []
                part_word_count = 0
                
                for part in sentence_parts:
                    part_words = len(part.split())
                    if part_word_count + part_words > max_chunk_size and current_part:
                        chunks.append(' '.join(current_part))
                        current_part = [part]
                        part_word_count = part_words
                    else:
                        current_part.append(part)
                        part_word_count += part_words
                
                if current_part:
                    chunks.append(' '.join(current_part))
            
            # Regular case - add sentence to current chunk if it fits
            elif current_word_count + sentence_word_count > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
                
        # Add any remaining text in the current chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    # Final check to ensure no empty chunks
    return [chunk for chunk in chunks if chunk.strip()]

def get_adaptive_pause_duration(prev_chunk, next_chunk, voice_id, base_duration=0.15):
    """Calculate an appropriate pause duration between chunks based on content and voice"""
    # Get voice characteristics
    voice_char = get_voice_characteristics(voice_id)
    pause_multiplier = voice_char['pause_multiplier']
    
    # Start with base duration
    pause_duration = base_duration
    
    # Check if we're transitioning between different types of sentences
    if prev_chunk and next_chunk:
        prev_type = detect_sentence_type(prev_chunk.split('.')[-1] if '.' in prev_chunk else prev_chunk)
        next_type = detect_sentence_type(next_chunk.split('.')[0] if '.' in next_chunk else next_chunk)
        
        # Longer pauses between different types of sentences
        if prev_type != next_type:
            pause_duration *= 1.5
        
        # Longer pauses after questions or exclamations
        if prev_type in ('question', 'exclamation'):
            pause_duration *= 1.3
        
        # Check for paragraph transitions (somewhat crude detection)
        if prev_chunk.endswith('.') and next_chunk[0].isupper():
            pause_duration *= 1.7
    
    # Apply voice-specific multiplier
    return pause_duration * pause_multiplier

def clean_memory():
    """Force garbage collection and clear CUDA cache if available"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_speech(text, voice_name, speed):
    """
    Generate speech from text using Kokoro TTS.
    
    Args:
        text (str): Input text to convert to speech
        voice_name (str): Selected voice name from the dropdown
        speed (float): Speech speed (0.5 to 2.0)
    
    Returns:
        str: Path to the generated audio file
    """
    start_time = time.time()
    
    # Handle empty input
    if not text or not text.strip():
        print("Empty input received")
        # Return a short silent audio file
        empty_audio = np.zeros(int(0.5 * 24000), dtype=np.float32)
        output_file = os.path.join(AUDIO_OUTPUT_DIR, f"empty_{int(time.time())}.wav")
        sf.write(output_file, empty_audio, 24000)
        return output_file
    
    # Clean and prepare text
    text = clean_text(text)
    
    # Ensure speed is within valid range
    speed = max(0.5, min(2.0, speed))
    
    # Get the voice ID from the display name
    voice_id = voices[voice_name]
    
    # Create a progress bar
    progress = gr.Progress()
    progress(0, desc="Preparing text...")
    
    # Generate a unique ID for this output
    output_id = str(uuid.uuid4())[:8]
    output_file = os.path.join(AUDIO_OUTPUT_DIR, f"tts_{output_id}.wav")
    
    # For long text, split into chunks with smaller size
    chunks = split_text_into_chunks(text, max_chunk_size=50)  # Use even smaller chunks
    total_chunks = len(chunks)
    
    print(f"Text split into {total_chunks} chunks")
    
    # Process each chunk and collect audio segments
    all_audio = []
    total_processed = 0
    missed_chunks = []
    
    # Function to save intermediate results in case of crashes
    def save_intermediate(audio_segments, step):
        if not audio_segments:
            return
        try:
            combined = np.concatenate(audio_segments)
            temp_file = os.path.join(AUDIO_OUTPUT_DIR, f"temp_{output_id}_{step}.wav")
            sf.write(temp_file, combined, 24000)
            print(f"Saved intermediate file at step {step}: {temp_file}")
        except Exception as e:
            print(f"Error saving intermediate file: {e}")
    
    prev_chunk = None
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        progress(i/total_chunks, desc=f"Processing chunk {i+1}/{total_chunks}")
        print(f"Processing chunk {i+1}/{total_chunks} ({len(chunk.split())} words): {chunk[:30]}...")
        
        # Check if we need to clean memory
        if i > 0 and i % 10 == 0:
            clean_memory()
        
        # Save intermediate results every 5 chunks
        if i > 0 and i % 5 == 0:
            save_intermediate(all_audio, i)
        
        # Generate cache key for this chunk
        cache_key = get_cache_key(chunk, voice_id, speed)
        cached_audio = check_cache(cache_key)
        
        if cached_audio is not None:
            # Use cached audio
            print(f"✓ Using cached audio for chunk {i+1}")
            all_audio.append(cached_audio)
            total_processed += 1
            
            # Add adaptive pause based on content and voice
            if i < total_chunks - 1:
                next_chunk = chunks[i+1] if i+1 < len(chunks) else None
                pause_duration = get_adaptive_pause_duration(chunk, next_chunk, voice_id)
                all_audio.append(np.zeros(int(pause_duration * 24000), dtype=np.float32))
            
            prev_chunk = chunk
            continue
        
        retry_count = 0
        max_retries = 2  # Try up to 3 times (original + 2 retries)
        success = False
        
        while not success and retry_count <= max_retries:
            try:
                # Process chunk with the pipeline
                audio_segments = []
                chunk_to_process = chunk
                
                # For retries, try with even smaller chunks or simplify text
                if retry_count > 0:
                    print(f"  Retry {retry_count}/{max_retries} for chunk {i+1}")
                    if retry_count == 1:
                        # First retry: simplify any complex punctuation
                        chunk_to_process = re.sub(r'[^\w\s.,!?-]', '', chunk_to_process)
                    elif retry_count == 2:
                        # Second retry: split into smaller parts
                        subchunks = chunk_to_process.split('.')
                        for j, subchunk in enumerate(subchunks):
                            if not subchunk.strip():
                                continue
                                
                            try:
                                for _, _, subchunk_audio in pipeline(subchunk.strip() + '.', voice=voice_id, speed=speed):
                                    if subchunk_audio is not None and len(subchunk_audio) > 0:
                                        all_audio.append(subchunk_audio)
                                        print(f"  ✓ Subchunk {j+1}/{len(subchunks)} processed")
                                        break
                            except Exception as sub_e:
                                print(f"  ⚠️ Error processing subchunk: {str(sub_e)}")
                        
                        # Add adaptive pause
                        if i < total_chunks - 1:
                            next_chunk = chunks[i+1] if i+1 < len(chunks) else None
                            pause_duration = get_adaptive_pause_duration(chunk, next_chunk, voice_id)
                            all_audio.append(np.zeros(int(pause_duration * 24000), dtype=np.float32))
                        
                        success = True  # We've handled this whole chunk through subchunks
                        break  # Exit the retry loop
                
                if retry_count <= 1:  # Only try normal processing for original and first retry
                    generator = pipeline(chunk_to_process, voice=voice_id, speed=speed)
                    
                    for _, _, chunk_audio in generator:
                        if chunk_audio is not None and len(chunk_audio) > 0:
                            audio_segments.append(chunk_audio)
                    
                    # Check if we got any audio for this chunk
                    if audio_segments:
                        # Concatenate all segments for this chunk
                        chunk_combined = np.concatenate(audio_segments)
                        
                        # Save to cache for future use
                        save_to_cache(cache_key, chunk_combined)
                        
                        all_audio.append(chunk_combined)
                        
                        # Add adaptive pause between chunks based on content
                        if i < total_chunks - 1:
                            next_chunk = chunks[i+1] if i+1 < len(chunks) else None
                            pause_duration = get_adaptive_pause_duration(chunk, next_chunk, voice_id)
                            all_audio.append(np.zeros(int(pause_duration * 24000), dtype=np.float32))
                        
                        total_processed += 1
                        chunk_duration = len(chunk_combined) / 24000
                        total_duration = sum(len(a) for a in all_audio) / 24000
                        
                        print(f"✓ Chunk {i+1} processed: {chunk_duration:.2f}s, Total: {total_duration:.2f}s")
                        success = True
                    else:
                        print(f"⚠️ No audio generated on try {retry_count+1}")
                        retry_count += 1
                
            except Exception as e:
                print(f"❌ Error on try {retry_count+1} for chunk {i+1}: {str(e)}")
                retry_count += 1
                
                # Brief pause before retry
                time.sleep(0.5)
        
        if not success:
            missed_chunks.append(i+1)
            print(f"⚠️ Failed to process chunk {i+1} after all retries")
        
        prev_chunk = chunk
    
    progress(0.9, desc="Finalizing audio...")
    clean_memory()  # Clean up memory before final processing
    
    # Concatenate all audio segments
    if all_audio:
        try:
            combined_audio = np.concatenate(all_audio)
            total_duration = len(combined_audio) / 24000
            print(f"Final audio duration: {total_duration:.2f} seconds")
            print(f"Processed {total_processed}/{total_chunks} chunks successfully")
            
            if missed_chunks:
                print(f"Missed chunks: {missed_chunks}")
        except Exception as e:
            print(f"Error concatenating audio: {e}")
            # Try to save whatever we can
            combined_audio = np.concatenate(all_audio[:len(all_audio)//2])
            print(f"Saving partial audio (first half): {len(combined_audio)/24000:.2f} seconds")
    else:
        # Fallback for empty results
        print("No audio was generated, creating empty audio")
        combined_audio = np.zeros(int(1.0 * 24000), dtype=np.float32)
    
    progress(0.95, desc="Saving audio file...")
    
    # Save the audio to a file
    sf.write(output_file, combined_audio, 24000)  # Kokoro uses 24kHz sample rate
    
    # Calculate total processing time
    elapsed_time = time.time() - start_time
    
    progress(1.0, desc="Complete!")
    print(f"Audio saved to {output_file}, duration: {len(combined_audio) / 24000:.2f} seconds")
    print(f"Total processing time: {elapsed_time:.1f} seconds")
    
    return output_file

# Create the Gradio interface
with gr.Blocks(title="Kokoro TTS") as interface:
    gr.Markdown("# Kokoro Text-to-Speech")
    gr.Markdown("Convert any text to natural-sounding speech with the Kokoro TTS model")
    
    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Enter Text", 
                placeholder="Type or paste text to convert to speech...", 
                lines=10, 
                max_lines=100,
                info="Supports large texts with automatic processing"
            )
            
            with gr.Accordion("Advanced Options", open=False):
                cache_checkbox = gr.Checkbox(
                    label="Use cache", 
                    value=True,
                    info="Store processed audio segments to speed up repeat processing"
                )
                
                use_adaptive_pauses = gr.Checkbox(
                    label="Use adaptive pauses", 
                    value=True,
                    info="Dynamically adjust pauses between sentences for natural speech"
                )
                
                clear_cache_btn = gr.Button("Clear Cache")
                
                def clear_cache():
                    try:
                        for cache_file in CACHE_DIR.glob("*.npy"):
                            cache_file.unlink()
                        return "Cache cleared successfully"
                    except Exception as e:
                        return f"Error clearing cache: {str(e)}"
                
                cache_status = gr.Textbox(label="Cache Status", interactive=False)
                clear_cache_btn.click(fn=clear_cache, outputs=cache_status)
        
        with gr.Column(scale=2):
            voice_dropdown = gr.Dropdown(
                label="Select Voice",
                choices=list(voices.keys()),
                value='🇺🇸 🚺 Heart ❤️',
                info="Choose from American English or British English voices"
            )
            
            speed_slider = gr.Slider(
                label="Speech Speed", 
                minimum=0.5, 
                maximum=2.0, 
                step=0.1, 
                value=1.0,
                info="1.0 is normal speed, lower is slower, higher is faster"
            )
            
            generate_btn = gr.Button("Generate Speech", variant="primary")
    
    audio_output = gr.Audio(label="Generated Speech")
    
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, voice_dropdown, speed_slider],
        outputs=audio_output,
    )
    
    gr.Markdown("""
    ## Tips for Best Results
    - For best results with very long text, consider adding proper punctuation
    - The system automatically breaks text into smaller chunks for processing
    - The app uses adaptive pauses between sentences for more natural-sounding speech
    - To process faster on repeat conversions, enable caching in Advanced Options
    - If you hear abrupt transitions, try adjusting the speed slightly
    """)
    
    with gr.Accordion("About", open=False):
        gr.Markdown("""
        This application uses the Kokoro TTS model to convert text to speech.
        
        **Features:**
        - Processes text of any length by intelligently chunking
        - Supports multiple voices in American and British English
        - Uses adaptive pauses between sentences based on content
        - Caches processed audio for faster repeated conversions
        - Automatically handles edge cases and errors
        
        The app includes advanced text preprocessing to improve pronunciation of
        abbreviations, numbers, and special characters.
        """)

# Launch the interface
interface.launch()