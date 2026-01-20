dhc/ # 项目根目录（强直+IBD医疗大模型RAG系统）
├── data # 数据总目录
│   ├── processed # 处理后数据目录（清洗、标注后的结构化数据）
│   ├── rag # RAG知识库目录（存放医疗文本，供检索增强使用）
│   └── raw # 原始数据目录（未经处理的医疗指南、论文等原始文档）
├── model # 模型相关目录
│   ├── lora_checkpoints # LoRA微调权重目录（存放微调过程中的 checkpoint 文件）
│   ├── qwen-7b-local # Qwen-7B本地模型目录（脱离网络依赖，可直接加载）
│   │   ├── config.json # 模型核心配置文件（定义模型结构、参数）
│   │   ├── configuration_qwen.py # Qwen模型专属配置类（支撑模型初始化）
│   │   ├── cpp_kernels.py # 模型加速CPP内核文件（优化运行效率）
│   │   ├── generation_config.json # 模型生成配置文件（定义生成长度、温度等）
│   │   ├── model-00001-of-00008.safetensors # 模型权重分片1（共8个，合计约13GB）
│   │   ├── model-00002-of-00008.safetensors # 模型权重分片2
│   │   ├── model-00003-of-00008.safetensors # 模型权重分片3
│   │   ├── model-00004-of-00008.safetensors # 模型权重分片4
│   │   ├── model-00005-of-00008.safetensors # 模型权重分片5
│   │   ├── model-00006-of-00008.safetensors # 模型权重分片6
│   │   ├── model-00007-of-00008.safetensors # 模型权重分片7
│   │   ├── model-00008-of-00008.safetensors # 模型权重分片8
│   │   ├── modeling_qwen.py # Qwen模型核心结构实现文件（定义网络层、注意力机制）
│   │   ├── model.safetensors.index.json # 权重分片索引文件（管理分片对应关系）
│   │   ├── qwen_generation_utils.py # 模型生成辅助工具类（自定义解码策略）
│   │   ├── qwen.tiktoken # Qwen模型专属分词器词典（支撑中文分词编码）
│   │   ├── tokenization_qwen.py # 分词器核心实现文件（文本与token转换）
│   │   └── tokenizer_config.json # 分词器配置文件（定义分词规则、最大长度）
│   └── scripts # 脚本目录（存放模型测试、RAG构建等执行脚本）
│       ├── hf_cache # Hugging Face冗余缓存目录（已无用，可删除释放空间）
│       │   └── models--Qwen--Qwen-7B # Qwen模型缓存子目录
│       │       ├── blobs # 缓存原始数据（哈希命名的底层文件）
│       │       │   ├── 0ced56cc3265fe03ee73658f845dc23c713389b5d68087e918d24d0e2dea624f # 缓存文件
│       │       │   ├── 173570d81fcb2390c057e31b3b5bfa7b45088824 # 缓存文件
│       │       │   ├── 2a526d66c3fc0779cb469fb9c838864ad2453d60 # 缓存文件
│       │       │   ├── 3dedac66034371aa3b284a7886e9ce0fde9245ebac60f507f089b33ef82a2912 # 缓存文件
│       │       │   ├── 45c0d16ac5e62a5441bca314cb1676c59e17880f # 缓存文件
│       │       │   ├── 4e8e1d8cadcb50ee9dffecc629d368371d268e88 # 缓存文件
│       │       │   ├── 59a22cb822f9e6d0a6a8415a7bab7b8448ce9672fc71f09266db264afc26ce48 # 缓存文件
│       │       │   ├── 7bc05473a78ade06d526cc206e4a01722563abe99099367cdcbb9b3bf670a5de # 缓存文件
│       │       │   ├── 81b25b14a58b62300d11b16c66933a8b631400bf846b27a2a5c5344629cd26e8 # 缓存文件
│       │       │   ├── 9b9b0e0416d84d7c88333eb261c77e5fe2d7f7be # 缓存文件
│       │       │   ├── 9dfd6266bcf80de9c3e5cd4e60300d839d03e459e48975b08d3e3b286044a306 # 缓存文件
│       │       │   ├── a7c2261f7223655d419e7f576c13a4c041d224a7 # 缓存文件
│       │       │   ├── b0999d47ea087bf79a075ed10889aa3497caff2356200204b032fba208109517 # 缓存文件
│       │       │   ├── d2b2922555bf1ac5850d94f6d63d7215d1139a9a # 缓存文件
│       │       │   ├── d9cee703ae23284f63078d8a15aca6d7c5614fdc # 缓存文件
│       │       │   ├── e61126f7ad8c520112c49808a73d59bee18c6da7693d9a963a4867f389c91e4a # 缓存文件
│       │       │   ├── f796c399c9e4616e72f0889f165d594fe0118b8a # 缓存文件
│       │       │   └── f8fe2cb434cefda404c506d541959e2fefc86884 # 缓存文件
│       │       ├── refs # 版本引用目录
│       │       │   └── main # 主版本引用标识
│       │       └── snapshots # 模型快照目录（软链接指向blobs文件）
│       │           └── ef3c5c9c57b252f3149c1408daf4d649ec8b6c85 # 快照版本目录（软链接集合）
│       │               ├── config.json -> ../../blobs/a7c2261f7223655d419e7f576c13a4c041d224a7 # 软链接（已冗余）
│       │               ├── configuration_qwen.py -> ../../blobs/f8fe2cb434cefda404c506d541959e2fefc86884 # 软链接（已冗余）
│       │               ├── cpp_kernels.py -> ../../blobs/d9cee703ae23284f63078d8a15aca6d7c5614fdc # 软链接（已冗余）
│       │               ├── generation_config.json -> ../../blobs/f796c399c9e4616e72f0889f165d594fe0118b8a # 软链接（已冗余）
│       │               ├── model-00001-of-00008.safetensors -> ../../blobs/9dfd6266bcf80de9c3e5cd4e60300d839d03e459e48975b08d3e3b286044a306 # 软链接（已冗余）
│       │               ├── model-00002-of-00008.safetensors -> ../../blobs/3dedac66034371aa3b284a7886e9ce0fde9245ebac60f507f089b33ef82a2912 # 软链接（已冗余）
│       │               ├── model-00003-of-00008.safetensors -> ../../blobs/81b25b14a58b62300d11b16c66933a8b631400bf846b27a2a5c5344629cd26e8 # 软链接（已冗余）
│       │               ├── model-00004-of-00008.safetensors -> ../../blobs/59a22cb822f9e6d0a6a8415a7bab7b8448ce9672fc71f09266db264afc26ce48 # 软链接（已冗余）
│       │               ├── model-00005-of-00008.safetensors -> ../../blobs/e61126f7ad8c520112c49808a73d59bee18c6da7693d9a963a4867f389c91e4a # 软链接（已冗余）
│       │               ├── model-00006-of-00008.safetensors -> ../../blobs/0ced56cc3265fe03ee73658f845dc23c713389b5d68087e918d24d0e2dea624f # 软链接（已冗余）
│       │               ├── model-00007-of-00008.safetensors -> ../../blobs/b0999d47ea087bf79a075ed10889aa3497caff2356200204b032fba208109517 # 软链接（已冗余）
│       │               ├── model-00008-of-00008.safetensors -> ../../blobs/7bc05473a78ade06d526cc206e4a01722563abe99099367cdcbb9b3bf670a5de # 软链接（已冗余）
│       │               ├── modeling_qwen.py -> ../../blobs/45c0d16ac5e62a5441bca314cb1676c59e17880f # 软链接（已冗余）
│       │               ├── model.safetensors.index.json -> ../../blobs/173570d81fcb2390c057e31b3b5bfa7b45088824 # 软链接（已冗余）
│       │               ├── qwen_generation_utils.py -> ../../blobs/4e8e1d8cadcb50ee9dffecc629d368371d268e88 # 软链接（已冗余）
│       │               ├── qwen.tiktoken -> ../../blobs/9b9b0e0416d84d7c88333eb261c77e5fe2d7f7be # 软链接（已冗余）
│       │               ├── tokenization_qwen.py -> ../../blobs/2a526d66c3fc0779cb469fb9c838864ad2453d60 # 软链接（已冗余）
│       │               └── tokenizer_config.json -> ../../blobs/d2b2922555bf1ac5850d94f6d63d7215d1139a9a # 软链接（已冗余）
│       └── qwen_download_test.py # 模型测试脚本（验证本地Qwen-7B加载及推理功能）
├── docs # 项目文档目录
│   ├── README.md # 项目说明文档（目录注释、运行指南、后续规划等）
│   ├── 项目目录说明.md # 项目目录结构说明文档（详细介绍每个子目录的作用）
