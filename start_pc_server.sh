#!/bin/bash
set -e
set -u

echo "$(date): 开始启动vLLM服务..."

export PYTHONHASHSEED=123456

# 确保目录存在并清理
mkdir -p /home/externals/suanfabu/data_pc
rm -rf /home/externals/suanfabu/data_pc/*

# 检查模型是否存在
if [ ! -d "/home/models/DeepSeek-R1-Distill-Qwen-32B" ]; then
    echo "错误: 模型目录不存在!"
    exit 1
fi

vllm serve /home/models/DeepSeek-R1-Distill-Qwen-32B \
--served-model-name DeepSeek-R1-Distill-Qwen-32B \
--max-model-len 131000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 8090 \
--block-size 128 \
--no-enable-prefix-caching \
--kv-transfer-config \
'{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "ucm_connector_name": "UcmNfsStore",
        "ucm_connector_config": {
            "storage_backends": "/home/externals/suanfabu/data_pc",
            "transferStreamNumber":16
        }
    }
}'
