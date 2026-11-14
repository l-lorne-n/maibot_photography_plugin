# Host 图片嵌入接口

摄影插件 0.4.1 通过 MaiBot 的 `image_embedding` 任务调用 `qwen3-vl-embedding`。插件将真实图片转换为向量，用来查找历史作品；VLM 再同时观看当前图和历史图，结合日记比较构图、色彩与视觉风格。

## 适配基线

- 开发实例：Host 1.4.3，SDK 2.22.1，MaiBot 本地基线提交 `e53915075`。
- 主程序需要提供接受 `image_base64` 的 `llm.embed` 能力。
- 已有文本专用实现需要使用 [host-image-embedding.patch](../patches/host-image-embedding.patch)。补丁只修改 `src/plugin_runtime/capabilities/core.py`，复用宿主已有的 `EmbeddingServiceClient.embed_image` 和模型指纹解析服务。
- 仅版本号相同不保证代码相同；应用前检查目标实现和补丁上下文。已支持该协议的 Host 无须重复应用。

## 应用补丁

在自己的 MaiBot 源码目录中执行，将示例路径替换为实际插件仓库路径：

```powershell
git status --short
git apply --check <插件仓库路径>/patches/host-image-embedding.patch
git apply <插件仓库路径>/patches/host-image-embedding.patch
```

先备份该文件，并确认已有修改与补丁相容。`--check` 失败时检查差异，不强制套用。完成修改后重启当前 bot；插件热加载不会重新导入已加载的 Host 核心能力。

本地适配时记录的文件 SHA-256：

| 文件状态 | SHA-256 |
| --- | --- |
| 补丁前 | `3ddc915bc4b6dd146b159817e817ce86c890fa3daff9d60e26be54cf1f2247ee` |
| 补丁后 | `450bcb3db7cbb9fd561e38220e195a34b8ddb66bbed53efe16bd9ec1444310e0` |

换行符变化也会改变哈希。哈希用于核对这份适配基线，实际应用仍以源码差异与检查结果为准。

## 能力协议

```python
result = await ctx.llm.embed(
    task_name="image_embedding",
    image_base64=base64.b64encode(image_bytes).decode("ascii"),
    mime_type="image/jpeg",
    preprocess_version="photography-embedding-jpeg1024-v1",
)
```

输入为单张图的 Base64，或带 MIME 的 data URI；解码后最多 2 MiB。图片输入与 `text`、`texts`、`input`、`inputs` 互斥。解码交给工作线程，原有文本/文本批量分支保留。

返回包含 `success`、`embedding`、`model_name`、`model_identifier`、`api_provider`、`mime_type`、`preprocess_version`、`request_protocol_hash` 和 `embedding_fingerprint`。指纹记录模型、供应商、维度与协议等向量空间信息。失败返回具体错误，不以文本嵌入替代图片嵌入。

SDK 默认嵌入任务是 `embedding`，图片调用必须显式指定 `image_embedding`。这里使用任务名，不直接传模型库条目名绕过任务分配。

## 配置与验证

1. 在模型库配置支持图片输入的 `qwen3-vl-embedding`，然后把 `image_embedding` 任务指向它。
2. 若使用 Alibaba / DashScope，宿主已有图片嵌入客户端会按供应商配置选择原生多模态协议；具体 URL 和密钥仍保存在 Host 的私有模型配置里。
3. 重启后，在摄影后台为现有收藏建立视觉记忆，检查模型名、向量数量和任务结果。
4. 再运行一轮摄影探索，检查精看观察中的历史对照。向量空间不同的记录不会混合检索。

插件在图片模型变化后按实际模型指纹分开缓存，不把不同维度或协议的向量直接相比较。本版开发实例实际输出为 2560 维；程序以返回结果及指纹为准。

## 代码与许可范围

插件运行代码、单元测试与 UI 位于仓库根目录，采用根目录 MIT 许可证。`patches/host-image-embedding.patch` 是针对 MaiBot 主程序的差异，来源基线为 [MaiBot](https://github.com/Mai-with-u/MaiBot)，按 `patches/LICENSE` 保留其 GNU GPLv3 许可。补丁不包含配置凭据、主程序完整副本或本机业务数据。
