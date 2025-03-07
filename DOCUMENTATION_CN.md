# Docker Setup for InsTaG Training Framework

## English

This pull request provides a complete Docker-based environment for the InsTaG training framework. It addresses several setup challenges documented in the issues by providing a consistent, containerized environment.

### Key Features:

1. **Dual Container Architecture:**
   - Main container (CUDA 11.7, Python 3.9) for training and inference
   - Separate Sapiens container (CUDA 12.1, Python 3.10) for geometry priors

2. **Helper Scripts:**
   - `docker-run.sh` - Simplifies common operations
   - `setup-docker.sh` - Automates initial setup and dependency installation

3. **Comprehensive Documentation:**
   - Complete workflow examples
   - Detailed troubleshooting guidance
   - Support for different audio feature extractors (DeepSpeech, Wav2Vec, AVE, HuBERT)

4. **Automated Setup:**
   - OpenFace integration for facial AU extraction
   - EasyPortrait model download
   - Sapiens model download

5. **Workflow Improvements:**
   - No manual environment conflicts
   - Simplified audio feature extraction
   - Streamlined teeth mask generation
   - Container-based geometry prior generation

The documentation includes examples for both short-video adaptation (with geometry priors) and long-video training, making it easier to use the framework in various scenarios.

---

## 中文

此 Pull Request 为 InsTaG 训练框架提供了完整的基于 Docker 的环境。它通过提供一致的容器化环境解决了 issues 中记录的几个设置挑战。

### 主要特点：

1. **双容器架构：**
   - 主容器（CUDA 11.7，Python 3.9）用于训练和推理
   - 单独的 Sapiens 容器（CUDA 12.1，Python 3.10）用于几何先验生成

2. **辅助脚本：**
   - `docker-run.sh` - 简化常见操作
   - `setup-docker.sh` - 自动化初始设置和依赖安装

3. **全面的文档：**
   - 完整的工作流示例
   - 详细的故障排除指南
   - 支持不同的音频特征提取器（DeepSpeech、Wav2Vec、AVE、HuBERT）

4. **自动化设置：**
   - OpenFace 集成用于面部 AU 提取
   - EasyPortrait 模型下载
   - Sapiens 模型下载

5. **工作流改进：**
   - 没有手动环境冲突
   - 简化的音频特征提取
   - 简化的牙齿遮罩生成
   - 基于容器的几何先验生成

文档包括短视频适应（带几何先验）和长视频训练的示例，使框架在各种场景中更易于使用。 