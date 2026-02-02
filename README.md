<div align="center">
 👋 Hi, everyone! 
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://seed.bytedance.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/5793e67c-79bb-4a59-811a-fcc7ed510bd4">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

<div align="center">
  <img src="./overview.png" alt="plot" height="400">
</div>

# A Predictive and Generative Solution for Liquid Electrolyte Formulation

<p align="center">
  <a href="https://huggingface.co/ByteDance-Seed/bamboo_mixer">
    <img src="https://img.shields.io/badge/Bamboo_mixer-Hugging Face-orange"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/badge/Apache License, Version 2.0-blue"></a>
</p>

We are extremely delighted to release **Bamboo-Mixer**. **Bamboo-Mixer** introduces a unified method to not only **predict electrolyte properties** accurately such as conductivity and anion ratio, but also allows **conditional generation** of electrolyte formulation based on target properties.

This innovation provides a generative workflow to **design molecular mixture** beyond electrolyte. The generated samples from the workflow are tested and proved further with experimental validation..

## Updates

- Feb 2025: Citation and README updated.
- Jan 2025: Cleanup codebase for improved readability and better open source&data.
- Aug 2025: First public release. Use this tagged version (`v0.0.1`) to reproduce the results reported in our manuscript.

## Branches
- `paper`: The first publicly released and archived branch with tagged version (`v0.0.1`). This branch matches the data and trained model checkpoint open-sourced on huggingface. This branch is recommended if you would like to use the pretrained checkpoint.
- `main`: The current-and-future working branch, with significant cleanups made on the `paper` branch. This branch uses the data included in the `data_oss` folder and does not use anything from huggingface. This branch is recommended if you would like to modify the code, retrain the entire framework, or make pull-requests.


## Data preparation

There are two different types of data: [`MonoData`](./formula_design/data/data.py#L243) and [`FormulaData`](./formula_design/data/data.py#L289)

- [`MonoData`](./formula_design/data/data.py#L243): single molecule data with molecular property labels.
- [`FormulaData`](./formula_design/data/data.py#L289): Molecular mixture data for liquid electrolytes with conductivity and anion ratio information.

More details about how to prepare data for different stages of training and inference can be found in [Data Preparation Instructions](./scripts/prepare_data/README.md).

## Training

**Predictive model (Three-stage training):**

1. Molecular pretraining: Multi-task learning of molecular property predictions.
2. Computational pretraining: Prediction of formulation property from MD simulation.
3. Experimental fine-tuning: Fine-tuning with experimental conductivity data.

**Generative model (Dual-stage training):**

1. Conditional diffusion: conditional DDPM to generate electrolyte embedding.
2. Decoder and distance matching: decode electrolyte embedding and obtain formulation.

See [Training Instructions](./scripts/train/README.md) for more details.

## Inference and Generation

**Molecular property prediction (11 molecular properties):**

- Tm: melting point (K)
- bp: boiling point (K)
- nD: refractive index (unitless)
- nD_liquid: refractive index (liquid, unitless)
- pka_a: pKa (unitless)
- pka_b: pKb (unitless)
- dc: dielectric constant (unitless)
- ST: surface tension (mN/m)
- density: mass density (g/cm^3)
- vis: viscosity (cP)
- vapP: vapor pressure (Pa)

**Predictive model (2 electrolyte properties):**

- Conductivity (mS/cm).
- Anion ratio (unitless, between 0 and 1).

**Generative model (3 generated files):**

- Electrolyte embedding.
- BoM vector of formulation.
- Corresponding conductivity and anion ratio.

See [Inference Instructions](./scripts/test_results/README.md) for more details.

## Data availability

The datasets and checkpoints can be found on HuggingFace: [huggingface/bamboo_mixer](https://huggingface.co/ByteDance-Seed/bamboo_mixer)

## Contributing

We welcome contributions to Bamboo-Mixer! If you have suggestions or improvements, please refers to `CONTRIBUTING.md`

## License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Citation

If you find Bamboo-Mixer useful for your research and applications, feel free to give us a star ⭐ or cite us using:

```bibtex
@article{yang2025unifiedpredictivegenerativesolution,
	author = {Yang, Zhenze and Wu, Yifan and Han, Xu and Zhang, Ziqing and Lai, Haoen and Mu, Zhenliang and Zheng, Tianze and Liu, Siyuan and Pu, Zhichen and Wang, Zhi and Yu, Zhiao and Gong, Sheng and Yan, Wen},
	date = {2026/01/28},
	doi = {10.1038/s42256-025-01173-w},
	isbn = {2522-5839},
	journal = {Nature Machine Intelligence},
	title = {A unified predictive and generative solution for liquid electrolyte formulation},
	url = {https://doi.org/10.1038/s42256-025-01173-w},
	year = {2026}
}
```

## About [ByteDance Seed Team](https://seed.bytedance.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.

<!-- 注释：About ByteDance Seed Team可直接复制使用 -->
