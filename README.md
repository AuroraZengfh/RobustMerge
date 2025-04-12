# CoPA-Merging


### Model Preparation

Create 'model' folder, donwload base model [LLaVA](https://github.com/haotian-liu/LLaVA) and put the checkpoint in the folder.

### Data and Instruction Preparation

Create `datasets` folder and download all dataset needed for merging.
Create `instructions` folder and download all dataset needed for merging.


For datasets and instructions listed in the paper, you can find seen datasets and instructions:
- **ScienceQA, ImageNet, Grounding, OCRVQA** in [CoIN](https://huggingface.co/datasets/Zacks-Chen/CoIN).
- **VQAv2, VizWiz, Flickr30k, IconQA** in [HiDe-LLaVA](github).

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
