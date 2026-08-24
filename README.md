# Setup env
修改原仓库中的setup
```
conda create -n llamauav python=3.10 -y
conda activate llamauav
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## Install Llama-UAV model
```
git clone git@github.com:HuangTH96/TravelUAV.git
cd TravelUAV/Model/Llama-UAV
pip install -e .
```

## dependency: flash-attn
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.5.9.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## airsim
```
pip install airsim==1.8.1 --no-build-isolation
```

## others
```
cd ~/TravelUAV
pip install -r requirements.txt
```

## fix bug: msgpack-rpc-python as mentioned [here](https://github.com/microsoft/AirSim/issues/3333#issuecomment-827894198)
```
pip uninstall msgpack-rpc-python -y
git clone https://github.com/tbelhalfaoui/msgpack-rpc-python.git
cd msgpack-rpc-python
git checkout fix-msgpack-dep
python setup.py install
cd ..
```

然后需要修改 AirSim 中的 `client.py`. \
首先查找 `client.py` 文件位置: \
```python -c "import airsim, os; print(os.path.join(os.path.dirname(airsim.__file__), 'client.py'))"``` \
打开文件后，将 \
```self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout=timeout_value, pack_encoding='utf-8', unpack_encoding='utf-8')``` \
改成： \
```self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout=timeout_value)```

# Pretrained weights for models


# Dataset