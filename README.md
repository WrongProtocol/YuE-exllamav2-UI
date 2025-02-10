## note this is an out of date sandbox to play with YuE-exlammav2 with gradio, running in a docker, with some small tweaks for the sake of testing (such as launching multiple consecutive  renders with the same settings).    I most likely will not be maintaining this fork or prepping it for a merge, as it will be significanlty out of date and the changes were either too specific to my use, or are easy enough to re-implement in a future version without dealing with dozens of merges. 

# YuE-Exllamav2-UI

Welcome to **YuE-Exllamav2-UI**, the ultimate optimized interface for music generation using YuE models with **ExLlamaV2 acceleration**. This project delivers the best possible performance for YuE models, achieving exceptional speed and efficiency on modern NVIDIA GPUs like the RTX 4090 and RTX 3060.

> **Note**: This project is a fork of the fork made by user sgsdxzy which has been optimized for performance using exllamav2 from the [YuE-exllamav2 repository](https://github.com/sgsdxzy/YuE-exllamav2). For more official information about the model, follow the [official repository](https://github.com/multimodal-art-projection/YuE).

---

## Interface


![Gradio Interface Preview](/preview.png)

---

## Key Features

- **ExLlamaV2 Optimization**: Inference acceleration with **Flash Attention 2** and **BF16 support**.
- **Exceptional Performance**: Up to **500% speedup** compared to the original implementation.
- **NVIDIA GPU Support**: Compatible with CUDA 12.2 and optimized for modern GPUs.
- **User-Friendly Interface**: Web-based Gradio interface for easy configuration and music generation.
- **Audio Playback and Download**: Listen to generated audio and download it directly from the interface.
- **Pre-configured Docker**: Easy deployment with Docker and support for persistent volumes.

---

## News and Updates


> **Note**: Keep an eye on updates to know when you need to do a docker pull alissonpereiraanjos/yue-exllamav2-interface

* **2025.02.04 🔥**: Added field to save the result file with a custom name
* **2025.02.03 🔥**: Added ExLlamaV2 integration with up to 500% speedup by sgsdxzy (https://github.com/sgsdxzy/YuE-exllamav2)
* **2025.02.01 (Oficial Repository) 🧑🏻‍💻 Inference Update**: 1. Support dual-track ICL mode. 2. Fix "instrumental" naming bug in output files. 3. Support seeding. 
* **2025.01.31 🔥**: Added possibility to select existing models using a dropdown, added Refresh File Explorer button and Warnings are ignored in the log.
- **2025.01.30 🔥**: Initial release with BF16 model support.


---

## Performance Benchmarks

### RTX 4090 24GB
| Stage         | Original | ExLlamaV2 | Speedup |
|---------------|----------|-----------|---------|
| Stage 1       | 282s     | 125s      | 2.25x   |
| Stage 2       | 666s     | 49s       | 13.6x   |
| **Total**     | 948s     | 174s      | **5.45x** |

*Configuration: BF16 models, Flash Attention 2 enabled, CFG enabled, batch size 4, 64k cache.*

### RTX 3060 Mobile 6GB
| Stage         | ExLlamaV2 |
|---------------|-----------|
| Stage 1       | 317s      |
| Stage 2       | 350s      |
| **Total**     | 667s      |

*Configuration: Quantized models with Q4/Q8 cache, Flash Attention 2 enabled.*

---

## How to Use

### Prerequisites

### Docker

Ensure you have Docker installed on your system. Follow the official Docker installation guides for your platform:

- [Install Docker on Windows](https://docs.docker.com/desktop/windows/install/)
- [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

### NVIDIA GPU Support

This interface **requires NVIDIA GPUs** for acceleration. Ensure you have the necessary hardware and drivers set up.

1. **Linux**:
   - Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
   - Ensure your NVIDIA drivers are properly installed and up to date.

2. **Windows/macOS**:
   - Refer to the respective Docker and NVIDIA documentation for GPU passthrough (e.g., WSL2 on Windows).
   - **Note**: GPU support is mandatory. Without compatible NVIDIA GPUs, the container will not function correctly.

### Docker Compose

To simplify the setup and management of the YuE-Exllamav2-UI, you can use Docker Compose. Docker Compose allows you to define and run multi-container Docker applications with a single configuration file (`docker-compose.yml`). Below are the steps to get started.

> **Note**: This **docker-compose.yml** file already exists in the root of the repository, you just need to download or copy the file and replace the directory mapping and run the command in the same directory as the file, see the explanation below.

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  yue-exllamav2:
    image: alissonpereiraanjos/yue-exllamav2-interface:latest
    container_name: yue-exllamav2
    restart: unless-stopped
    ports:
      - "7860:7860"
    environment:
      - DOWNLOAD_MODELS=all_bf16
    volumes:
      - /path/to/models:/workspace/models
      - /path/to/outputs:/workspace/outputs
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

Run the container:

```bash
docker-compose up -d
```

**Access the Interface**: Once the container is running, access the Gradio UI at `http://localhost:7860`.

### Explanation of the Configuration

- **`image`**: Specifies the Docker image to use (`alissonpereiraanjos/yue-exllamav2-interface:latest`).
- **`container_name`**: Sets a name for the container (`yue-interface`).
- **`restart: unless-stopped`**: Ensures the container restarts automatically unless manually stopped.
- **`ports`**: Maps container ports to the host:
  - `7860:7860`: Port for accessing the Gradio UI.
  - `8888:8888`: Optional additional port (JupyterLab).
- **`environment`**: Defines environment variables:
  - `DOWNLOAD_MODELS=all`: Downloads all available models. Replace `all` with specific model keys (e.g., `YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot`) to download only selected models.
- **`volumes`**: Maps host directories to the container:
  - `/path/to/models:/workspace/models`: Directory where models will be stored.
  - `/path/to/outputs:/workspace/outputs`: Directory where generated outputs will be saved.
  - Replace `/path/to/models` and `/path/to/outputs` with the actual paths on your system.
- **`deploy.resources.reservations.devices`**: Enables GPU support in the container (requires NVIDIA GPU and drivers).

### Customization

- **Specific models**: To download only specific models, modify the `DOWNLOAD_MODELS` environment variable in the `docker-compose.yml` file. For example:

  ```yaml
  environment:
    - DOWNLOAD_MODELS=YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot
  ```

- **Different ports**: If you need to use different ports, adjust the port mappings under the `ports` section.
---

### Direct Docker Run

To run without Docker Compose:

```bash
docker run --gpus all -d \
  -p 7860:7860 \
  -e DOWNLOAD_MODELS=all_bf16 \
  -v /path/to/models:/workspace/models \
  -v /path/to/outputs:/workspace/outputs \
  alissonpereiraanjos/yue-exllamav2-interface:latest
```

- `--gpus all`: Enables NVIDIA GPU support.
- `-d`: Runs the container in detached mode (background).
- `-p 7860:7860`: Exposes port `7860` for accessing the Gradio UI at [http://localhost:7860](http://localhost:7860).
- `-p 8888:8888`: Exposes port `8888` for additional services if applicable.
- `-e DOWNLOAD_MODELS=all_bf16`: Downloads all bf16 available models upon initialization.

## Environment Variables

- **DOWNLOAD_MODELS**: Determines which models to download.
  - Set to `all_bf16` to download all available models (BF16).
  - Alternatively, specify a comma-separated list of model keys to download specific models (e.g., `DOWNLOAD_MODELS=YuE-s2-1B-general,YuE-s1-7B-anneal-en-cot`).


---

## Supported Models

| Model Key                         | Model HF Repository                             | Container Path                                      | Quantization |
|-----------------------------------|----------------------------------------------|----------------------------------------------------|--------------|
| `xcodec_mini_infer`               | [`m-a-p/xcodec_mini_infer`](https://huggingface.co/m-a-p/xcodec_mini_infer) | `/workspace/YuE-Interface/inference/xcodec_mini_infer` | N/A          |
| `YuE-s1-7B-anneal-en-cot`         | [`m-a-p/YuE-s1-7B-anneal-en-cot`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot) | `/workspace/models/YuE-s1-7B-anneal-en-cot`       | BF16         |
| `YuE-s1-7B-anneal-en-icl`         | [`m-a-p/YuE-s1-7B-anneal-en-icl`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl) | `/workspace/models/YuE-s1-7B-anneal-en-icl`       | BF16         |
| `YuE-s1-7B-anneal-jp-kr-cot`      | [`m-a-p/YuE-s1-7B-anneal-jp-kr-cot`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-cot) | `/workspace/models/YuE-s1-7B-anneal-jp-kr-cot`    | BF16         |
| `YuE-s1-7B-anneal-jp-kr-icl`      | [`m-a-p/YuE-s1-7B-anneal-jp-kr-icl`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-icl) | `/workspace/models/YuE-s1-7B-anneal-jp-kr-icl`    | BF16         |
| `YuE-s1-7B-anneal-zh-cot`         | [`m-a-p/YuE-s1-7B-anneal-zh-cot`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-cot) | `/workspace/models/YuE-s1-7B-anneal-zh-cot`       | BF16         |
| `YuE-s1-7B-anneal-zh-icl`         | [`m-a-p/YuE-s1-7B-anneal-zh-icl`](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-icl) | `/workspace/models/YuE-s1-7B-anneal-zh-icl`       | BF16         |
| `YuE-s2-1B-general`               | [`m-a-p/YuE-s2-1B-general`](https://huggingface.co/m-a-p/YuE-s2-1B-general) | `/workspace/models/YuE-s2-1B-general`             | BF16         |
| `YuE-upsampler`                   | [`m-a-p/YuE-upsampler`](https://huggingface.co/m-a-p/YuE-upsampler) | `/workspace/models/YuE-upsampler`                 | BF16         |

---

### Mapping Directories for Models and Output

You can mount host directories to store models and outputs outside the container:

```bash
docker run --gpus all -it \
  -v /path/to/models:/workspace/models \
  -v /path/to/outputs:/workspace/outputs \
  -p 7860:7860 \
  -p 8888:8888 \
  -e DOWNLOAD_MODELS=false \
  alissonpereiraanjos/yue-exllamav2-interface:latest
```

- `-v /path/to/models:/workspace/models`: Mounts the host's `/path/to/models` directory to `/workspace/models` inside the container.
- `-v /path/to/outputs:/workspace/outputs`: Mounts the host's `/path/to/outputs` directory to `/workspace/outputs` inside the container.
- `-e DOWNLOAD_MODELS=false`: Skips automatic model downloads (useful if models are already present in the mounted directories).

## RunPod Deployment

If you prefer to use **RunPod**, you can quickly deploy an instance based on this image by using the following template link:

[**Deploy on RunPod**](https://runpod.io/console/deploy?template=nah47krmt0&ref=8t518hht)

This link directs you to the RunPod console, allowing you to set up a machine directly with the YuE Interface image. Configure your GPU, volume mounts, and environment variables as needed.

**Tip**: If you generate music frequently, consider creating a **Network Volume** in RunPod. This allows you to store models and data persistently, avoiding repeated downloads and saving time.

**Recommended Settings:**
- GPU: RTX 4090 or A100
- Network Volume: Minimum 100GB for model storage
- Environment Variables: `DOWNLOAD_MODELS=all_bf16`

## Update Docker Image (Important)

To update the Docker image with the latest changes, run:

```bash
docker pull alissonpereiraanjos/yue-exllamav2-interface:latest
```

**Note**: Always update the image before running the container to ensure you have the latest features and fixes. This is especially important when deploying on RunPod, as it pulls the latest image upon creating a new pod.

### Model Suffixes Explained

The suffixes in the model keys indicate specific training or optimization techniques applied to the models:

| Suffix | Meaning               | Description                                                                                     |
|--------|-----------------------|-------------------------------------------------------------------------------------------------|
| `COT`  | **Chain-of-Thought**  | Models trained with *Chain-of-Thought* to enhance reasoning and logical generation capabilities.|
| `ICL`  | **In-Context Learning** | Models optimized for *In-Context Learning*, allowing dynamic adaptation based on the provided context.|

**Examples:**

- `YuE-s1-7B-anneal-en-cot`: A model trained with *Chain-of-Thought* techniques.
- `YuE-s1-7B-anneal-en-icl`: A model optimized for *In-Context Learning*.

## Acknowledgements

A special thanks to the [YuE-exllamav2](https://github.com/sgsdxzy/YuE-exllamav2) repository for their incredible optimization work, which made this project possible.

---

## Support

For technical support or questions:
- [GitHub Issues](https://github.com/alissonpereiraanjos/YuE-Exllamav2-UI/issues)
- [CivitAI Profile](https://civitai.com/user/alissonerdx)

---

## Citation

If you use this project in your research, please consider citing:

```bibtex
@misc{yuan2025yue,
  title={YuE: Open Music Foundation Models for Full-Song Generation},
  author={Ruibin Yuan et al.},
  year={2025},
  howpublished={\url{https://github.com/multimodal-art-projection/YuE}}
}
```

---

**Experience music generation with accelerated performance!** 🎵🚀

---
