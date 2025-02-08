import json
import os
import subprocess
import threading
import queue
import signal
import gradio as gr
import time
import tempfile
import glob
import sys
import re
import shutil  # Added to copy files
from collections import OrderedDict

# TODO
# Choose use SDPA or FlashAttention2
# Choose profiles for differents gpus
# Apply transforms patch


# -------------------------------------------------
# If you are using Conda, set these paths accordingly
CONDA_ACTIVATE_PATH = "/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV_NAME = "pyenv"
BASE_REPO_DIR = "/workspace/YuE-exllamav2-UI"
BASE_YUE_DIR = "/workspace/YuE-exllamav2-UI/src/yue"
BASE_MODELS_DIR = "/workspace/models"
BASE_OUTPUTS_DIR = "/workspace/outputs"
BASE_INPUTS_DIR = "/workspace/inputs"
# Default Hugging Face models
DEFAULT_STAGE1_MODEL = f"{BASE_MODELS_DIR}/YuE-s1-7B-anneal-en-cot"
DEFAULT_STAGE2_MODEL = f"{BASE_MODELS_DIR}/YuE-s2-1B-general"
TOKENIZER_MODEL = f"{BASE_YUE_DIR}/mm_tokenizer_v0.2_hf/tokenizer.model"

sys.path.append(os.path.join(f"{BASE_YUE_DIR}", 'xcodec_mini_infer'))
sys.path.append(os.path.join(f"{BASE_YUE_DIR}", 'xcodec_mini_infer', 'descriptaudiocodec'))

# Output directory
os.makedirs(BASE_OUTPUTS_DIR, exist_ok=True)

os.makedirs(BASE_INPUTS_DIR, exist_ok=True)

with open(f"{BASE_REPO_DIR}/prompt_egs/lyrics.txt", mode='r') as lyrics_example_file:
    lyrics_example = lyrics_example_file.read()

with open(f"{BASE_REPO_DIR}/prompt_egs/genre.txt", mode='r') as genre_example_file:
    genre_example = genre_example_file.read()

# -------------------------------------------------

# Functions to List and Categorize Models
def get_models(model_dir):
    """
    Lists all models in the specified directory and categorizes them as Stage1, Stage2, or both.
    """
    if not os.path.isdir(model_dir):
        return [], [], []
    
    # List directories only
    models = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]
    stage1_models = []
    stage2_models = []
    both_stage_models = []

    for model in models:
        lower_name = model.lower()
        model_path = os.path.join(model_dir, model)
        if 's1' in lower_name:
            stage1_models.append(model_path)
        if 's2' in lower_name:
            stage2_models.append(model_path)
        if 's1' not in lower_name and 's2' not in lower_name:
            both_stage_models.append(model_path)
    return stage1_models, stage2_models, both_stage_models

def get_quantization_type(model_path):
    """
    Determines the quantization type based on the model's name.
    """
    basename = os.path.basename(model_path).lower()
    if '-int4' in basename:
        return 'int4'
    elif '-nf4' in basename:
        return 'nf4'
    elif '-int8' in basename:
        return 'int8'
    else:
        return 'bf16'

# Model Directory
stage1_models, stage2_models, both_stage_models = get_models(BASE_MODELS_DIR)
stage1_choices = stage1_models + both_stage_models
stage2_choices = stage2_models + both_stage_models

# Queues for logs and audio paths
log_queue = queue.Queue()
audio_path_queue = queue.Queue()

process_dict = {}
process_lock = threading.Lock()

def load_and_process_genres(json_path):
    """
    Loads JSON data, processes genres, timbres, genders, moods, and instruments,
    removes duplicates (case insensitive), and returns a sorted list of unique values.
    """
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Combine all relevant categories into a single list
    categories = ['genre', 'timbre', 'gender', 'mood', 'instrument']
    all_items = [item.strip() for category in categories for item in data.get(category, [])]
    
    # Use a set for deduplication (case insensitive)
    unique_items = OrderedDict()
    for item in all_items:
        key = item.lower()
        if key not in unique_items and item:  # Skip empty strings
            unique_items[key] = item
    
    # Sort alphabetically while preserving original capitalization
    sorted_items = sorted(unique_items.values(), key=lambda x: x.lower())
    
    return sorted_items

js = """
function createLink() {
    let baseUrl = window.location.origin;
    baseUrl = baseUrl.replace("7860", "8080");
    const tagLink = `${baseUrl}/repo/wav_top_200_tags.json`;
    document.getElementById("tags_link").href = tagLink;
}
"""


custom_log_box_css = """
#log_box textarea {
    overflow-y: scroll;
    max-height: 400px;  /* Set a max height for the log box */
    white-space: pre-wrap;  /* Preserve line breaks and white spaces */
    border: 1px solid #ccc;
    padding: 10px;
    font-family: monospace;
    scrollbar-width: thin!important;
}"""

def get_selected_file(file_paths):
    """
    Handles file selection and prepares it for download.
    """
    # Handle the case when file_paths is a string (single file)
    if isinstance(file_paths, str):
        if os.path.isdir(file_paths):
            return None, "Please select a single file and not a folder."
        if not os.path.exists(file_paths):
            return None, f"File not found: {file_paths}"
        if not file_paths.lower().endswith('.mp3'):
            return None, f"File is not in .mp3 format: {file_paths}"
        return file_paths, f"File '{os.path.basename(file_paths)}' ready for download."

    # Handle the case when file_paths is a list (multiple files)
    if isinstance(file_paths, list) and file_paths:
        file_path = file_paths[0]  # Use the first file
        if os.path.isdir(file_path):
            return None, "Please select a single file and not a folder."
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"
        if not file_path.lower().endswith('.mp3'):
            return None, f"File is not in .mp3 format: {file_paths}"
        return file_path, f"File '{os.path.basename(file_paths)}' ready for download."

    return None, "Invalid or no file selected."

def read_subprocess_output(proc, log_queue, audio_path_queue):
    """Reads subprocess stdout line by line, placing them into log_queue and audio_path_queue."""
    for line in iter(proc.stdout.readline, b''):
        decoded_line = line.decode("utf-8", errors="replace")
        print(f"Subprocess output: {decoded_line}")  # Debugging
        log_queue.put(decoded_line)
        
        # Detect the line containing "Successfully created '"
        if "Created mix:" in decoded_line:
            # Extract the audio path using regex
            match = re.search(r"Created mix:\s*([^']+\.mp3)", decoded_line)
            if match:
                audio_path = match.group(1)
                print(f"Audio path found: {audio_path}")  # Debugging
                audio_path_queue.put(audio_path)
    proc.stdout.close()
    proc.wait()
    with process_lock:
        if proc.pid in process_dict:
            del process_dict[proc.pid]

def stop_generation(pid):
    """Send signals to stop the subprocess if running."""
    if pid is None:
        return "No process is running."
    with process_lock:
        proc = process_dict.get(pid)
    if not proc:
        return "No process found or it has already stopped."

    if proc.poll() is not None:
        return "Process already finished."

    try:
        # Send SIGTERM first
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(2)
        if proc.poll() is None:
            # If still running, force kill
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with process_lock:
            if pid in process_dict:
                del process_dict[pid]
        return "Inference stopped successfully."
    except Exception as e:
        return f"Error stopping process: {str(e)}"
    
def generate_song(
    stage1_model,
    stage1_model_quantization,
    stage2_model,
    stage2_model_quantization,
    tokenizer_model,
    genre_txt_path,
    lyrics_txt_path,
    run_n_segments,
    stage2_batch_size,
    output_dir,
    cuda_idx,
    max_new_tokens,
    seed,
    use_audio_prompt,
    audio_prompt_file,
    prompt_start_time,
    prompt_end_time,
    use_dual_tracks_prompt,
    vocal_track_prompt_file,
    instrumental_track_prompt_file,
    prompt_start_time_2,
    prompt_end_time_2,
    disable_offload_model,
    keep_intermediate,
    custom_filename,
    stage1_cache_size,
    stage1_cache_mode,
    stage2_cache_size,
    stage2_cache_mode
    # use_mmgp,
    # mmgp_profile,
    # use_sdpa,
    # use_torch_compile,
    # use_transformers_patch
):
    """Spawns infer.py to generate music, capturing logs in real time."""
    os.makedirs(output_dir, exist_ok=True)

    # If using an audio prompt, copy the file to DEFAULT_INPUT_DIR
    if use_audio_prompt and audio_prompt_file is not None:
        # Check if audio_prompt_file is a valid path
        if isinstance(audio_prompt_file, str):
            audio_filename = os.path.basename(audio_prompt_file)
            # Replace all special characters with '_' to avoid issues with the command
            audio_filename = re.sub(r"[^a-zA-Z0-9.]", "_", audio_filename)
            saved_audio_path = os.path.join(BASE_INPUTS_DIR, audio_filename)
            shutil.copy(audio_prompt_file, saved_audio_path)
        else:
            return "Invalid audio prompt file format.", None
    else:
        saved_audio_path = ""
        
        
    if use_dual_tracks_prompt and vocal_track_prompt_file is not None and instrumental_track_prompt_file is not None:
        if isinstance(vocal_track_prompt_file, str):
            vocal_track_filename = os.path.basename(vocal_track_prompt_file)
            vocal_track_filename = re.sub(r"[^a-zA-Z0-9.]", "_", vocal_track_filename)
            saved_vocal_track_path = os.path.join(BASE_INPUTS_DIR, vocal_track_filename)
            shutil.copy(vocal_track_prompt_file, saved_vocal_track_path)
        else:
            return "Invalid vocal track prompt file format.", None
        
        if isinstance(instrumental_track_prompt_file, str):
            instrumental_track_filename = os.path.basename(instrumental_track_prompt_file)
            # Replace all special characters with '_' to avoid issues with the command
            instrumental_track_filename = re.sub(r"[^a-zA-Z0-9.]", "_", instrumental_track_filename)
            saved_instrumental_track_path = os.path.join(BASE_INPUTS_DIR, instrumental_track_filename)
            shutil.copy(instrumental_track_prompt_file, saved_instrumental_track_path)
        else:
            return "Invalid instrumental track file format.", None
    else:
        saved_vocal_track_path = ""
        saved_instrumental_track_path = ""

    # Build base command with '-u' for unbuffered output
    cmd = [
        "python", "-u", f"{BASE_YUE_DIR}/infer.py",  # Added '-u' here
        "--stage1_use_exl2",
        "--stage1_model", f"'{stage1_model}'",
        "--stage1_cache_size", str(stage1_cache_size),
        "--stage1_cache_mode", f"'{stage1_cache_mode}'",
        # "--quantization_stage1", f"{stage1_model_quantization}",
        "--stage2_use_exl2",
        "--stage2_model", f"'{stage2_model}'",
        "--stage2_cache_size", str(stage2_cache_size),
        "--stage2_cache_mode", f"'{stage2_cache_mode}'",
        # "--quantization_stage2", f"{stage2_model_quantization}",
        #"--tokenizer", f"'{tokenizer_model}'",
        "--genre_txt", f"'{genre_txt_path}'",
        "--lyrics_txt", f"'{lyrics_txt_path}'",
        "--run_n_segments", str(run_n_segments),
        "--stage2_batch_size", str(stage2_batch_size),
        "--output_dir", f"'{output_dir}'",
        "--cuda_idx", str(cuda_idx),
        "--seed", f"{seed}",
        "--max_new_tokens", str(max_new_tokens),
        "--basic_model_config", f"'{BASE_YUE_DIR}/xcodec_mini_infer/final_ckpt/config.yaml'",
        "--resume_path", f"'{BASE_YUE_DIR}/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth'",
        "--config_path", f"'{BASE_YUE_DIR}/xcodec_mini_infer/decoders/config.yaml'",
        "--vocal_decoder_path", f"'{BASE_YUE_DIR}/xcodec_mini_infer/decoders/decoder_131000.pth'",
        "--inst_decoder_path", f"'{BASE_YUE_DIR}/xcodec_mini_infer/decoders/decoder_151000.pth'"
    ]
    
    if custom_filename.strip():
        cmd.append(f"--custom_filename '{custom_filename}'")
    
    if use_audio_prompt and saved_audio_path:
        cmd += [
            "--use_audio_prompt",
            "--audio_prompt_path", f"'{saved_audio_path}'",
            "--prompt_start_time", str(prompt_start_time),
            "--prompt_end_time", str(prompt_end_time)
        ]
        
    if use_dual_tracks_prompt and saved_vocal_track_path and saved_instrumental_track_path:
        cmd += [
            "--use_dual_tracks_prompt",
            "--vocal_track_prompt_path", f"'{saved_vocal_track_path}'",
            "--instrumental_track_prompt_path", f"'{saved_instrumental_track_path}'",
            "--prompt_start_time", str(prompt_start_time_2),
            "--prompt_end_time", str(prompt_end_time_2)
        ]

    if disable_offload_model:
        cmd.append("--disable_offload_model")
    if keep_intermediate:
        cmd.append("--keep_intermediate")
        
    # If using conda, wrap the command
    if os.path.isfile(CONDA_ACTIVATE_PATH):
        prefix_cmd = (
            f"bash -c 'source {CONDA_ACTIVATE_PATH} && "
            f"conda activate {CONDA_ENV_NAME} && "
        )
        suffix_cmd = "'"
        final_cmd_str = prefix_cmd + " ".join(cmd) + suffix_cmd
    else:
        final_cmd_str = " ".join(cmd)
    
    print(final_cmd_str)
    
    proc = subprocess.Popen(
        final_cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid  # allows signal handling
    )

    with process_lock:
        process_dict[proc.pid] = proc

    # Thread to read logs
    thread = threading.Thread(target=read_subprocess_output, args=(proc, log_queue, audio_path_queue), daemon=True)
    thread.start()

    return f"Inference started. Outputs will be saved in {output_dir}...", proc.pid

def update_logs(current_logs):
    """Pull all new lines from log_queue and append to current_logs."""
    new_text = ""
    while not log_queue.empty():
        new_text += log_queue.get()
    return current_logs + new_text

def build_gradio_interface():
    theme = gr.themes.Base()
    with gr.Blocks(title="YuE Exllamav2: Open Full-song Generation Foundation Model", theme=theme, css=custom_log_box_css) as demo:
        gr.Markdown("# YuE Exllamav2 - Gradio Interface\nEnter your Genre and Lyrics, then generate & listen!")

        with gr.Column():
            
            # Replace Textboxes with Dropdowns for Automatic Model Selection
            stage1_model = gr.Dropdown(
                label="Stage1 Model",
                choices=stage1_choices,
                value=DEFAULT_STAGE1_MODEL,
                info="Select the checkpoint path for the Stage 1 model.",
                interactive=True
            )
            stage1_model_quantization = gr.Dropdown(
                choices=["bf16", "int8", "int4", "nf4"],
                label="Select the quantization of the Stage1 model",
                value=get_quantization_type(DEFAULT_STAGE1_MODEL),
                interactive=True
            )
            stage2_model = gr.Dropdown(
                label="Stage2 Model",
                choices=stage2_choices,
                value=DEFAULT_STAGE2_MODEL,
                info="Select the checkpoint path for the Stage 2 model.",
                interactive=True
            )
            stage2_model_quantization = gr.Dropdown(
                choices=["bf16", "int8", "int4", "nf4"],
                label="Select the quantization of the Stage2 model",
                value=get_quantization_type(DEFAULT_STAGE2_MODEL),
                interactive=True
            )
            
            # TODO: remove the tokenizer model
            tokenizer_model = gr.Textbox(
                label="Tokenizer Model",
                value=TOKENIZER_MODEL,
                info="Path to the model tokenizer.",
                visible=False
            )
            # gr.Markdown("#### Optimizations using MMGP (Memory Management for the GPU Poor) by DeepBeepMeep")
            # with gr.Row():
            #     with gr.Column():
            #         use_mmgp = gr.Checkbox(
            #             label="Use MMGP? (Only works with original BF16 model, Quantization will be performed based on the chosen profile.)",
            #             value=False,
            #             info="If set, Memory Management for GPU Poor by deepbeepmeep will be used."
            #         )
                    
            #         gr.Markdown(f"""
            #                     **MMGP Profile:**
            #                     - Profile 1: The fastest but requires 16 GB of VRAM.
            #                     - Profile 3: A bit slower and the model is quantized to 8 bits but requires 12 GB of VRAM.
            #                     - Profile 4: Very slow as this will incur sequencial offloading.
            #                     """)
                    
            #         mmgp_profile = gr.Dropdown(
            #             label="MMGP Profile",
            #             choices=[1, 3, 4],
            #             value=1,
            #             visible=False,
            #             interactive=True
            #         )
                    
            #         def toggle_mmgp_profile(checked):
            #             return gr.update(visible=checked)
                    
            #         use_mmgp.change(
            #             fn=toggle_mmgp_profile,
            #             inputs=use_mmgp,
            #             outputs=mmgp_profile
            #         )
                    
            #         use_transformers_patch = gr.Checkbox(
            #             label="Use Transformers Patch (optional)(< 10GB of VRAM)?",
            #             value=False,
            #             info="If set, the model will use the transformers patch (this patch overwrites two files from the transformers library, It may take a little longer to start generating after clicking the Generate button as a backup of the transformers will be performed or restored)."
            #         )
                    
            #         use_sdpa = gr.Checkbox(
            #             label="Use SDPA? (Can be used with MMGP Profile 4)",
            #             value=False,
            #             info="If set, the model will use SDPA instead of FlashAttention2."
            #         )
                    
            #         use_torch_compile = gr.Checkbox(
            #             label="Torch Compile? (Can be used with MMGP Profile 4)",
            #             value=False,
            #             info="If set, the model will be compiled using torch compile."
            #         )
            
            gr.Markdown(f"""
                        **Tips:**
                        1. `genres` should include details like instruments, genre, mood, vocal timbre, and vocal gender.
                        2. The length of `lyrics` segments and the `--max_new_tokens` value should be matched. For example, if `--max_new_tokens` is set to 3000, the maximum duration for a segment is around 30 seconds. Ensure your lyrics fit this time frame.
                        3. If using audio prompt，the duration around 30s will be fine.
                            
                        **Notice:**
                        1. A suitable [Genre] tag consists of five components: genre, instrument, mood, gender, and timbre. All five should be included if possible, separated by spaces. The values of timbre should include "vocal" (e.g., "bright vocal").

                        2. The order of the tags is flexible. For example, a stable genre control string might look like: "[Genre] inspiring female uplifting pop airy vocal electronic bright vocal vocal."

                        3. Additionally, we have introduced the "Mandarin" and "Cantonese" tags to distinguish between Mandarin and Cantonese, as their lyrics often share similarities.
                        """)
                   
            # Dropdowns for genre and lyrics
            genres = load_and_process_genres(f"{BASE_REPO_DIR}/top_200_tags.json")
            genre_select = gr.Dropdown(
                label="Select Music Genres",
                info="Select genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.",
                choices=genres,
                interactive=True,
                multiselect=True,
                max_choices=50
            )
            
            genre_textarea = gr.Textbox(
                visible=False,
            )
            
            genre_select.change(
                fn=lambda x: " ".join(x),
                inputs=genre_select,
                outputs=genre_textarea
            )
            
            lyrics_textarea = gr.Textbox(
                label="Lyrics Text",
                lines=4,
                placeholder="Type the lyrics here...",
                info="Text containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.",
                value=lyrics_example
            )

            run_n_segments = gr.Number(
                label="Number of Segments",
                value=4,
                precision=0,
                info="Set Number of Segments to the number of lyric sections if you want to generate a full song. Additionally, you can increase Stage2 Batch Size based on your available GPU memory."
            )
            stage1_cache_size = gr.Number(
                label="Stage1 Cache Size",
                value=16384,
                precision=0,
                info="The cache size used in Stage 1 inference."
            )
            
            stage1_cache_mode = gr.Dropdown(
                label="Stage1 Cache Mode",
                choices=["FP16", "Q8", "Q6", "Q4"],
                value="Q8",
                interactive=True,
                info="The cache mode used in Stage 1 inference (FP16, Q8, Q6, Q4). Quantized k/v cache will save VRAM at the cost of some speed and precision."
            )
            
            stage2_cache_size = gr.Number(
                label="Stage2 Cache Size (8192, 16384)",
                value=32768,
                precision=0,
                info="The cache size used in Stage 2 inference."
            )
            
            stage2_cache_mode = gr.Dropdown(
                label="Stage2 Cache Mode",
                choices=["FP16", "Q8", "Q6", "Q4"],
                value="FP16",
                interactive=True,
                info="The cache mode used in Stage 2 inference (FP16, Q8, Q6, Q4). Quantized k/v cache will save VRAM at the cost of some speed and precision."
            )
            
            stage2_batch_size = gr.Number(
                label="Stage2 Batch Size (4 is default)",
                value=20,
                precision=0,
                info="The batch size used in Stage 2 inference.",
            )
            output_dir = gr.Textbox(
                label="Output Directory",
                value=BASE_OUTPUTS_DIR,
                info="The directory where generated outputs will be saved."
            )
            cuda_idx = gr.Number(
                label="CUDA Index",
                value=0,
                precision=0
            )
            max_new_tokens = gr.Number(
                label="Max New Tokens",
                value=3000,
                precision=0,
                info="The maximum number of new tokens to generate in one pass during text generation."
            )
            
            disable_offload_model = gr.Checkbox(
                label="Disable Offload Model?",
                value=False,
                visible=False,
                info="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference."
            )
            
            keep_intermediate = gr.Checkbox(
                label="Keep Intermediate Files?",
                value=False,
                info="If set, intermediate outputs will be saved during processing."
            )
            
            
              
            gr.Markdown(f"""
If you want to use music in-context-learning (provide a reference song), enable `Use Audio Prompt?` and provide `Audio File`, `Prompt Start Time (s)`, and `Prompt Start Time (s)` to specify the audio segment. 

Note: 
- ICL requires a different ckpt, e.g. `m-a-p/YuE-s1-7B-anneal-en-icl`.

- Music ICL generally requires a 30s audio segment. The model will write new songs with similar style of the provided audio, and may improve musicality.

- Dual-track ICL works better in general, requiring both vocal and instrumental tracks.

- For single-track ICL, you can provide a mix, vocal, or instrumental track.

- You can separate the vocal and instrumental tracks using [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) or [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui).
                        """)

            use_audio_prompt = gr.Checkbox(
                label="Use Audio Prompt?",
                value=False,
                info="If set, the model will use an audio file as a prompt during generation."
            )
            audio_prompt_file = gr.File(
                label="Upload Audio Prompt",
                file_types=["audio"],
                visible=False,
                file_count="single",  # Ensure that only one file is uploaded
            )
            
            prompt_start_time = gr.Number(
                label="Prompt Start Time (s)",
                value=0,
                visible=False,
                info="The start time in seconds to extract the audio prompt from the given audio file."
            )
            prompt_end_time = gr.Number(
                label="Prompt End Time (s)",
                value=30,
                visible=False,
                info="The end time in seconds to extract the audio prompt from the given audio file."
            )
            
            def toggle_audio_prompt(checked):
                return [
                    gr.update(visible=checked),
                    gr.update(visible=checked),
                    gr.update(visible=checked)
                ]

            use_audio_prompt.change(
                fn=toggle_audio_prompt,
                inputs=use_audio_prompt,
                outputs=[audio_prompt_file, prompt_start_time, prompt_end_time]
            )
            
            use_dual_tracks_prompt = gr.Checkbox(
                label="Use Dual Tracks Prompt?",
                value=False,
                info="If set, the model will use an dual tracks files as a prompt during generation."
            )
            vocal_track_prompt_file = gr.File(
                label="Upload Vocal Track File",
                file_types=["audio"],
                visible=False,
                file_count="single",  # Ensure that only one file is uploaded
            )
            
            instrumental_track_prompt_file = gr.File(
                label="Upload Instrumental Track File",
                file_types=["audio"],
                visible=False,
                file_count="single",  # Ensure that only one file is uploaded
            )
            
            prompt_start_time_2 = gr.Number(
                label="Prompt Start Time (s)",
                value=0,
                visible=False,
                info="The start time in seconds to extract the audio prompt from the given audio file."
            )
            prompt_end_time_2 = gr.Number(
                label="Prompt End Time (s)",
                value=30,
                visible=False,
                info="The end time in seconds to extract the audio prompt from the given audio file."
            )
            
            def toggle_dual_track_prompt(checked):
                return [
                    gr.update(visible=checked),
                    gr.update(visible=checked),
                    gr.update(visible=checked),
                    gr.update(visible=checked),
                ]

            use_dual_tracks_prompt.change(
                fn=toggle_dual_track_prompt,
                inputs=use_dual_tracks_prompt,
                outputs=[vocal_track_prompt_file, instrumental_track_prompt_file, prompt_start_time_2, prompt_end_time_2]
            )
            
            def handle_checkbox_change(use_audio_prompt_checked, use_dual_tracks_prompt_checked):
                if use_audio_prompt_checked:
                    return True, False, *toggle_audio_prompt(True), *toggle_dual_track_prompt(False)
                elif use_dual_tracks_prompt_checked:
                    return False, True, *toggle_audio_prompt(False), *toggle_dual_track_prompt(True)
                else:
                    return False, False, *toggle_audio_prompt(False), *toggle_dual_track_prompt(False)
                
            use_audio_prompt.change(
                fn=handle_checkbox_change,
                inputs=[use_audio_prompt, use_dual_tracks_prompt],
                outputs=[
                    use_audio_prompt, use_dual_tracks_prompt,
                    audio_prompt_file, prompt_start_time, prompt_end_time,
                    vocal_track_prompt_file, instrumental_track_prompt_file, prompt_start_time_2, prompt_end_time_2
                ]
            )
    
            use_dual_tracks_prompt.change(
                fn=handle_checkbox_change,
                inputs=[use_audio_prompt, use_dual_tracks_prompt],
                outputs=[
                    use_audio_prompt, use_dual_tracks_prompt,
                    audio_prompt_file, prompt_start_time, prompt_end_time,
                    vocal_track_prompt_file, instrumental_track_prompt_file, prompt_start_time_2, prompt_end_time_2
                ]
            )
            
            seed = gr.Number(
                label="Seed",
                value=42,
                precision=0,
                info="Seed for random number generation."
            )

            custom_filename = gr.Textbox(
                label="Custom Save Filename (Optional)(Do not add .mp3 in the name)",
                value="",
                info="Custom filename for the generated output."
            )

            generate_button = gr.Button("Generate Music")
            stop_button = gr.Button("Stop", visible=False)
            
            # with gr.Row():
            #     total_steps = gr.State(0)
            #     steps_per_epoch = gr.State(0)
            #     current_epoch_display = gr.Textbox(label="Epoch Progress", interactive=False, value="Epoch: N/A")
            #     current_step_display = gr.Textbox(label="Step Progress", interactive=False, value="Step: N/A")

            log_box = gr.Textbox(
                label="Logs",
                value="",
                lines=20,
                max_lines=30,
                interactive=False,
                elem_id="log_box"
            )
            
            # workaround for the issue of the file explorer not updating
            def update_file_explorer():
                return gr.FileExplorer(root_dir=BASE_REPO_DIR)
            def update_file_explorer_2():
                return gr.FileExplorer(root_dir=BASE_OUTPUTS_DIR)
            
            explorer = gr.FileExplorer(root_dir=BASE_OUTPUTS_DIR, interactive=True, label="File Explorer", file_count="single", elem_id="file_explorer", glob="**/*.mp3", every=1)     
            update_button = gr.Button("Refresh File Explorer")
            
            update_button.click(update_file_explorer, outputs=explorer).then(update_file_explorer_2, outputs=explorer)
            
            with gr.Column():   
                gr.Markdown("### Select a single file from the file explorer for download.")
            
            audio_status = gr.Textbox(label="File Status", interactive=False)
             
            # Section to show audio and allow download
            audio_player = gr.Audio(
                label="Generated Audio",
                type="filepath",
                value=None,
                interactive=False
            )
            
            # Event: When a file is selected in the explorer
            explorer.change(
                fn=get_selected_file,
                inputs=[explorer],
                outputs=[audio_player, audio_status],
            )

        # Hidden states
        generation_pid = gr.State(None)
        current_audio_path = gr.State(None)

        # Adding Callbacks to Update Quantization Based on Selected Model
        stage1_model.change(
            fn=lambda model_path: get_quantization_type(model_path),
            inputs=stage1_model,
            outputs=stage1_model_quantization
        )

        stage2_model.change(
            fn=lambda model_path: get_quantization_type(model_path),
            inputs=stage2_model,
            outputs=stage2_model_quantization
        )

        def on_generate_click(
            stage1_model,
            stage1_model_quantization,
            stage2_model,
            stage2_model_quantization,
            tokenizer_model,
            genre_text,
            lyrics_text,
            run_n_segments,
            stage2_batch_size,
            output_dir,
            cuda_idx,
            max_new_tokens,
            seed,
            use_audio_prompt,
            audio_prompt_file,
            prompt_start_time,
            prompt_end_time,
            use_dual_tracks_prompt,
            vocal_track_prompt_file,
            instrumental_track_prompt_file,
            prompt_start_time_2,
            prompt_end_time_2,
            disable_offload_model,
            keep_intermediate,
            custom_filename,
            stage1_cache_size,
            stage1_cache_mode,
            stage2_cache_size,
            stage2_cache_mode
            # use_mmgp,
            # mmgp_profile,
            # use_sdpa,
            # use_torch_compile,
            # use_transformers_patch
        ):
            """Triggered when user clicks 'Generate Music'."""
            # Check if a process is already running
            with process_lock:
                if process_dict:
                    return (
                        "Another process is running. Please stop it before starting a new one.",
                        None,
                        gr.update(visible=True),
                        gr.update(visible=False),
                    )

            # Writes genre_text and lyrics_text to temporary .txt files
            def write_temp_file(content, suffix=".txt"):
                fd, path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                return path

            genre_tmp_path = write_temp_file(genre_text, ".txt")
            lyrics_tmp_path = write_temp_file(lyrics_text, ".txt")

            msg, pid = generate_song(
                stage1_model,
                stage1_model_quantization,
                stage2_model,
                stage2_model_quantization,
                tokenizer_model,
                genre_tmp_path,
                lyrics_tmp_path,
                run_n_segments,
                stage2_batch_size,
                output_dir,
                cuda_idx,
                max_new_tokens,
                seed,
                use_audio_prompt,
                audio_prompt_file,
                prompt_start_time,
                prompt_end_time,
                use_dual_tracks_prompt,
                vocal_track_prompt_file,
                instrumental_track_prompt_file,
                prompt_start_time_2,
                prompt_end_time_2,
                disable_offload_model,
                keep_intermediate,
                custom_filename,
                stage1_cache_size,
                stage1_cache_mode,
                stage2_cache_size,
                stage2_cache_mode
                # use_mmgp,
                # mmgp_profile,
                # use_sdpa,
                # use_torch_compile,
                # use_transformers_patch
            )
            # If the generation started successfully, hide "Generate" and show "Stop"
            if pid:
                return (msg, pid, gr.update(visible=False), gr.update(visible=True))
            else:
                return (msg, None, gr.update(visible=True), gr.update(visible=False))

        generate_button.click(
            fn=on_generate_click,
            inputs=[
                stage1_model,
                stage1_model_quantization,
                stage2_model,
                stage2_model_quantization,
                tokenizer_model,
                genre_textarea,
                lyrics_textarea,
                run_n_segments,
                stage2_batch_size,
                output_dir,
                cuda_idx,
                max_new_tokens,
                seed,
                use_audio_prompt,
                audio_prompt_file,
                prompt_start_time,
                prompt_end_time,
                use_dual_tracks_prompt,
                vocal_track_prompt_file,
                instrumental_track_prompt_file,
                prompt_start_time_2,
                prompt_end_time_2,
                disable_offload_model,
                keep_intermediate,
                custom_filename,
                stage1_cache_size,
                stage1_cache_mode,
                stage2_cache_size,
                stage2_cache_mode
                # use_mmgp,
                # mmgp_profile,
                # use_sdpa,
                # use_torch_compile,
                # use_transformers_patch
            ],
            outputs=[log_box, generation_pid, generate_button, stop_button]
        )

        def on_stop_click(pid):
            """Triggered when the user clicks 'Stop'."""
            status = stop_generation(pid)
            return (status, None, gr.update(visible=True), gr.update(visible=False))

        stop_button.click(
            fn=on_stop_click,
            inputs=[generation_pid],
            outputs=[log_box, generation_pid, generate_button, stop_button]
        )

        last_log_update = gr.State("")
        last_audio_update = gr.State(None)
        
        
        # audio_result_path = gr.Textbox(
        #     visible=False,
        #     interactive=False
        # )
        
        def refresh_state(log_text, pid, old_audio, last_log, last_audio):
            # Collect all new logs
            new_logs = ""
            while not log_queue.empty():
                new_logs += log_queue.get()
                
            # Collect new audio if available
            new_audio = old_audio
            while not audio_path_queue.empty():
                new_audio = audio_path_queue.get()

            # Check for real changes
            updated_log = log_text + new_logs if new_logs else log_text
            has_log_changes = updated_log != last_log
            has_audio_changes = new_audio != last_audio and new_audio is not None
            
            if has_audio_changes:
                final_path = os.path.join(BASE_OUTPUTS_DIR, os.path.basename(new_audio))
                audio_player.value = final_path
                audio_status.value = "File ready for download."
                #explorer.refresh()

            # Return gr.update() for fields that have changed, else no update
            return (
                updated_log if has_log_changes else gr.update(),  # log_box
                new_audio if has_audio_changes else gr.update(),  # current_audio_path
                updated_log if has_log_changes else last_log,     # last_log_update
                new_audio if has_audio_changes else last_audio    # last_audio_update
            )

        log_timer = gr.Timer(0.5, active=False)

        log_timer.tick(
            fn=refresh_state,
            inputs=[log_box, generation_pid, current_audio_path, last_log_update, last_audio_update],
            outputs=[log_box, current_audio_path, last_log_update, last_audio_update]
        )

        def activate_timer():
            return gr.update(active=True)

        generate_button.click(fn=activate_timer, outputs=[log_timer])
        stop_button.click(fn=activate_timer, outputs=[log_timer])
        
        def deactivate_timer():
            return gr.update(active=False)
        
        stop_button.click(fn=deactivate_timer, outputs=[log_timer])
        return demo

if __name__ == "__main__":
    interface = build_gradio_interface()
    # Adjust the port as needed
    _, base_url, _ = interface.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/workspace", ".", os.getcwd()])
