<p align="center">
  <a href="https://phenonet.org/">
    <img src="./assets/images/phenonet.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PhenoNet</h3>
  <p align="center">
    A two-stage lightweight deep learning framework for real-time wheat phenophase classification
    <br />
    <a href="https://phenonet.org/"><strong>Try the PhenoNet Platform »</strong></a>
    <br />
     <a href="https://help.phenonet.org/">Help Documents</a>
    ·
    <a href="https://datahub.aiphenomics.com/phenonet.html/">WheatPheno Dataset</a>
    ·
      <a href="https://github.com/Jinlab-AiPhenomics/PhenoNet/issues">Report Bug</a>
</p>

## 📝 Overview
Ruinan Zhang<sup>1</sup>, Shichao Jin<sup>1*</sup>, Yuanhao Zhang<sup>1</sup>, Jingrong Zang<sup>1</sup>, Yu Wang<sup>1</sup>, Qing Li<sup>1</sup>, Zhuangzhuang Sun<sup>1</sup>, Xiao Wang<sup>1</sup>, Qin Zhou<sup>1</sup>, Jian Cai<sup>1</sup>, Shan Xu<sup>1</sup>, Yanjun Su<sup>2</sup>, Jin Wu<sup>3</sup>, Dong Jiang<sup>1</sup>

<sup>1</sup>Plant Phenomics Research Centre, Academy for Advanced Interdisciplinary Studies, Collaborative Innovation Centre for Modern Crop Production co-sponsored by Province and Ministry, College of Agriculture, Nanjing Agricultural University, Nanjing 210095, China

<sup>2</sup>State Key Laboratory of Vegetation and Environmental Change, Institute of Botany, Chinese Academy of Sciences, Beijing 100093, China

<sup>3</sup>School of Biological Sciences and Institute for Climate and Carbon Neutrality, The University of Hong Kong, Pokfulam Road, Hong Kong, China

\* Corresponding author: Shichao Jin ([jschaon@njau.edu.cn](mailto:jschaon@njau.edu.cn); OR jinshichao1993@gmail.com)

## ⚙️ Installation
### Install with Conda
1. Please install Anaconda firstly.

2. We recommend cloning the PhenoNet repository into a clear folder.

   ```python
   cd {your folder}
   git clone https://github.com/Jinlab-AiPhenomics/PhenoNet.git
   cd PhenoNet
   ```

3. Create a clear environment for PhenoNet and activate the environment.

   ```python
   conda create -n phenonet python=3.9
   conda activate phenonet
   ```
4. Install Python requirements.

   ```python
   pip install -r requirements.txt
   ```
## 🚀 Usage

### Architecture

1. Activate the environment.

   ```	python
   conda activate phenonet
   cd PhenoNet
   ```
   
2. Run predict.py with specified parameters.

   ```python
   python test.py
   ```

### Prediction

Please visit https://help.phenonet.org/ to access online training documents.

### Training

Please visit https://help.phenonet.org/ to access online training documents.

## 🙏 Acknowledgement

- [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models/tree/main)

## 📄 License

[GPL-3.0](LICENSE) © PhenoNet