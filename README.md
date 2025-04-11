# CoPA-Merging


### Data Preparation

Create `data` folder and download all dataset needed for merging.

For dataset listed in the paper, you can find seen datasets:
- **ScienceQA, ImageNet, Grounding, VQAv2, OCRVQA** in [CoIN](https://huggingface.co/datasets/Zacks-Chen/CoIN).
- **VizWiz, Flickr30k, IconQA** in [HiDe-LLaVA](github).

We also construct unseen datasets including **AOKVQA, ImageNet-R, Screen2W and TabMWP** in .

You can also formulate your custom data and place them in the folder.


### Training
Follow standard parameter-efficient fine-tuning procedure in [LLaVA](https://github.com/haotian-liu/LLaVA) to obtain individual checkpoints for each dataset.

### Merging Evaluation



```bibtex
@article{zeng2025parameter,
  title={Parameter efficient merging for multimodal large language models with complementary parameter adaptation},
  author={Zeng, Fanhu and Guo, Haiyang and Zhu, Fei and Shen, Li and Tang, Hao},
  journal={arXiv preprint arXiv:2502.17159},
  year={2025}
}
```



## Acknowledgememnt

The code is based on  [LLaVA](https://github.com/haotian-liu/LLaVA), [TIES-Merging](https://github.com/prateeky2806/ties-merging). Thanks for these great works and open sourcing! 

If you find them helpful, please consider citing them as well. 
