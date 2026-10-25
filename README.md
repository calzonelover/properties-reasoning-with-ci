# Exploring Extrinsic and Intrinsic Properties for Effective Reasoning with Code Interpreter

[![arXiv](https://img.shields.io/badge/arXiv-2510.01332-red?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.16934)

## Models and Datasets
The models and datasets are available at Huggingface

### Models

| Model | Link |
|:---|:---|
| `Qwen2.5-Coder-7B-Instruct-ReTool-w-CoBe` | [🤗 airesearch/Qwen2.5-Coder-7B-Instruct-ReTool-w-CoBe](https://huggingface.co/airesearch/Qwen2.5-Coder-7B-Instruct-ReTool-w-CoBe) |
| `Qwen2.5-7B-Instruct-ReTool-w-CoBe` | [🤗 airesearch/Qwen2.5-7B-Instruct-ReTool-w-CoBe](https://huggingface.co/airesearch/Qwen2.5-7B-Instruct-ReTool-w-CoBe) |
| `Qwen3-8B-ReTool-w-CoBe` | [🤗 airesearch/Qwen3-8B-ReTool-w-CoBe](https://huggingface.co/airesearch/Qwen3-8B-ReTool-w-CoBe) |
| `Qwen2.5-7B-Instruct-ReTool-Rewrite` | [🤗 airesearch/Qwen2.5-7B-Instruct-ReTool-Rewrite](https://huggingface.co/airesearch/Qwen2.5-7B-Instruct-ReTool-Rewrite) |


### Datasets

| Dataset | Link |
|:---|:---|
| `ReTool-SFT-CoBe` | [🤗 airesearch/ReTool-SFT-CoBe](https://huggingface.co/datasets/airesearch/ReTool-SFT-CoBe) |
| `ReTool-SFT-CoBe-Think` | [🤗 airesearch/ReTool-SFT-CoBe-Think](https://huggingface.co/datasets/airesearch/ReTool-SFT-CoBe-Think) |
| `ReTool-SFT-Rewrite` | [🤗 airesearch/ReTool-SFT-Rewrite](https://huggingface.co/datasets/airesearch/ReTool-SFT-Rewrite) |

## The Scripts

### Training-Time
- [Fine-Tuning](./train-time/train/README.md)
- [Evaluation](./train-time/evaluation/README.md)

### Test-Time Scaling
- [Inference](./test-time-scaling)

## Citation

```bib
@misc{payoungkhamdee2026exploringextrinsicintrinsicproperties,
      title={Exploring Extrinsic and Intrinsic Properties for Effective Reasoning with Code Interpreter}, 
      author={Patomporn Payoungkhamdee and Napat Laosaengpha and Jenta Wonglertsakul and Pittawat Taveekitworachai and Pume Tuchinda and Panjapong Poobanchuen and Ekapol Chuangsuwanich and Can Udomcharoenchaikit and Samuel Cahyawijaya and Peerat Limkonchotiwat and Sarana Nutanong},
      year={2026},
      eprint={2606.16934},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.16934}, 
}
```