# 1. Setup
1. setup conda env for CUDA VERSION: 13.0

    `./requirements.txt` 中列出了所需要的依赖包，但是在配置过程中，发现 `flash-attn`，`msgpack-rpc-python` 以及 `airsim` 会有问题，无法直接 `pip install -r requirements.txt`，因此，先手动安装这三个包，再使用 `pip install` 安装剩下的
    ```bash
    conda create -n llamauav python=3.10 -y
    conda activate llamauav
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    
    git clone git@github.com:HuangTH96/TravelUAV.git    # 会报错 git lfs 相关，不用管，后面会修复
    cd TravelUAV/Model/Llama-UAV
    pip install -e .
    # additional pkg: flash-attn
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    pip install flash_attn-2.5.9.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    # additional pkg: msgpakc-rpc-python
    cd <path-to-TravelUAV>
    git clone https://github.com/tbelhalfaoui/msgpack-rpc-python.git
    cd msgpack-rpc-python
    git checkout fix-msgpack-dep
    python setup.py install
    # additional pkgs: tornado, msgpack
    pip install msgpack tornado==4.5.3
    # airsim
    pip install airsim==1.8.1 --no-build-isolation
    # 查看 [1.1 fix issue with airsim]
    # other dependencies
    cd <path-to-TravelUAV>
    pip install -r requirements.txt
    ```
2. setup Llama_uav model
    ```bash
    pip install huggingface_hub
    cd <path-to-TravelUAV>/Model/LLaMA-UAV
    mkdir model_zoo

    export HF_ENDPOINT=https://hf-mirror.com

    # vicuna-7b-v1.5
    cd model_zoo && mkdir -p LLM/vicuna-7b-v1.5 LAVIS
    huggingface-cli download lmsys/vicuna-7b-v1.5 \
        --local-dir ./LLM/vicuna-7b-v1.5

    # EVA-ViT-G and QFormer-7b
    cd LAVIS
    wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
    wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth

    # checkpoint for LLaMA-UAV model
    mkdir ../../work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32 
    huggingface-cli download wangxiangyu0814/llama-uav-7b --local-dir ../../work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32

    mkdir ../../work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4
    huggingface-cli download wangxiangyu0814/traveluav-traj-model --local-dir ../../work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4
    ```
3. download dataset

    TravelUAV_dataset 包括两部分：
    1. 数据集元信息：TravelUAV_data_json GitHub 代码仓库里通过 git-lfs 追踪的部分元信息（.json） 文件的LFS 对象缺失。因此，clone下来的仓库中，没有相关的.json文件，也就是之前clone时出现的报错，就算git pull，也没用。需要从 Hugging Face 的 TravelUAV_data_json 仓库单独下载
    2. 数据集部分 就是[README](https://github.com/buaa-colalab/TravelUAV/blob/main/Model/LLaMA-UAV/README.md#dataset)中 *Download the dataset from here* 的连接
    ```bash
    # 首先查看相关的问题文件是什么。这里有三个文件： trainset.json, train_balance.json, 以及 GroundingDINO 的权重文件
    cd <path-to-TravelUAV>
    git lfs status | grep deleted

    # trainset.json
    cd <path-to-TravelUAV>
    huggingface-cli download wangxiangyu0814/TravelUAV_data_json data/uav_dataset/trainset.json --repo-type dataset --local-dir ./data/uav_dataset

    # train_balance.json
    huggingface-cli download wangxiangyu0814/TravelUAV_data_json data/traj_train/train_balance.json --repo-type dataset --local-dir .

    # weights for GroudingDINO
    cd ./src/model_wrapper/utils/GroundingDINO
    wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

    # download TravelUAV_dataset
    tmux new-session -t download_traveluav
    cd <path-to-TravelUAV>/.. && vim download.sh   # 查看[1.2 数据集下载脚本]
    chmod +x download.sh
    ./download.sh

    # unzip dataset 【注意：可能无法一次性解压所有压缩包，需要最后查一下是否所有压缩包都解压了】
    cd <path-to-TravelUAV_dataset>
    for f in *.zip; do 
        echo "解压 $f ..."
        7z x "$f" -o./extracted/
    done

    # 运行Python脚本，生成 merged_data.json 文件
    python <path-to-TravelUAV>/Model/LLaMA-UAV/tools/generate_merged_json.py \
            --root_dir <path-to-TravelUAV_dataset>/extracted

    # 运行Python脚本，将image转换成tensor
    python <path-to-TravelUAV>/Model/LLaMA-UAV/tools/preprocess_image2tensor.py \
            --root_dir <path-to-TravelUAV_dataset>/extracted
    ```
4. download envs
    ```bash
    tmux new-session -t download_env
    cd <path-to-TravelUAV>/.. && vim download_env.sh   # 查看[1.3 环境下载脚本]
    chmod +x download_env.sh
    ./download_env.sh

    # unzip
    cd <path-to-TravelUAV_env>
    for f in *.zip; do
        echo "解压 $f ..."
        7z x "$f" -o./extracted/
    done

    # Fix assert error of SEARCH_ENVs_PATH in AirVLNSimulatorServerTool.py 
    # 具体问题查看 AirVLNSimulatorServerTool.py 中有关该变量的 TODO
    cd ./extracted && mkdir envs
    ```
5. start simulator server and run closed loop simulation

    ```bash
    cd <path-to-TravelUAV>/airsim_plugin
    python AirVLNSimulatorServerTool.py --port 30000 --root_path <path-to-TravelUAV_env>/extracted

    cd <path-to-TravelUAV>  # 必须回到 TravelUAV 目录下，否则会报错

    # run Dagger NYC
    chmod +x ./scripts/dagger_NYC.sh # 另外需要修改 root_dir
    bash +x ./scripts/dagger_NYC.sh

    # or Eval
    chmod +x ./scripts/eval.sh
    bash ./scripts/eval.sh     # 需要修改 root_dir

    chmod +x ./scripts/metrics.sh
    bash ./scripts/metrics.sh
    ```


## 1.1 fix issue with airism
1. 查找 `client.py` 的位置 `python -c "import airsim, os; print(os.path.join(os.path.dirname(airsim.__file__), 'client.py'))"`
2. 将文件中的 `self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout=timeout_value, pack_encoding='utf-8', unpack_encoding='utf-8')` 改成 `self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout=timeout_value)`

## 1.2 数据集下载脚本
```bash
#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_DISABLE_XET=1   # 关键：绕开经常超时的 xet CDN，退回普通下载

MAX_RETRY=1000
i=0
until huggingface-cli download wangxiangyu0814/TravelUAV \
    --repo-type dataset \
    --local-dir /data/huangth/TravelUAV_dataset \
    --max-workers 1; do
    i=$((i+1))
    echo "下载中断，第 $i 次重试... $(date)"
    if [ $i -ge $MAX_RETRY ]; then
        echo "重试次数过多，退出"
        break
    fi
    sleep 10
done
echo "全部完成或已达最大重试次数"
```

## 1.3 环境下载脚本
```bash
#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_DISABLE_XET=1   # 关键：绕开经常超时的 xet CDN，退回普通下载

MAX_RETRY=1000
i=0
until huggingface-cli download wangxiangyu0814/TravelUAV_env \
    --repo-type dataset \
    --local-dir /data/huangth/TravelUAV_env \
    --max-workers 1; do
    i=$((i+1))
    echo "下载中断，第 $i 次重试... $(date)"
    if [ $i -ge $MAX_RETRY ]; then
        echo "重试次数过多，退出"
        break
    fi
    sleep 10
done
echo "全部完成或已达最大重试次数"

```