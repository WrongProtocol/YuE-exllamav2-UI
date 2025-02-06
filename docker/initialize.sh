#!/bin/bash

echo "pod started"



if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY > authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
fi

#!/bin/bash
if [[ -z "${HF_TOKEN}" ]] || [[ "${HF_TOKEN}" == "enter_your_huggingface_token_here" ]]
then
    echo "HF_TOKEN is not set"
else
    echo "HF_TOKEN is set, logging in..."
    huggingface-cli login --token ${HF_TOKEN}
fi

# Start nginx as reverse proxy to enable api access
service nginx start

# Start JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.allow_origin='*' --NotebookApp.token='' --ServerApp.preferred_dir=/workspace --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' &
echo "JupyterLab started"

mkdir -p /workspace

# Check if diffusion-pipe directory exists in /workspace
echo "Copying YuE-exllamav2-UI to /workspace..."
cp -rf /YuE-exllamav2-UI  /workspace/

# the initial image of this was meant to be turnkey, so it
# always grabbed an update of the code. but i dont want it
# to do that. 
#echo "Updating YuE-exllamav2-UI ..."
#cd /workspace/YuE-exllamav2-UI 
#git pull --verbose

cp -f /entrypoint.sh /workspace/entrypoint.sh

bash /workspace/entrypoint.sh

sleep infinity
