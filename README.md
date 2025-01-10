# ICM-plus
An evaluation of In-Context Matting by HUST-TinySmart and an expansion of ICM-57 dataset. Official original paper is shown below.

<p align="center">
<a href="https://arxiv.org/pdf/2403.15789.pdf"><img  src="demo/src/icon/arXiv-Paper.svg" ></a>
<!-- <a href="https://link.springer.com/article/"><img  src="demo/src/icon/publication-Paper.svg" ></a> -->
<a href="https://opensource.org/licenses/MIT"><img  src="demo/src/icon/license-MIT.svg"></a>

</p>


<h4 align="center">This is an implementation of the IconMatting Model from the paper <a href="https://arxiv.org/abs/2403.15789">In-Context Matting</a>.</h4>

<h4 align="center">Details of the model architecture can be found in <a href="https://tiny-smart.github.io/icm.github.io/">TinySmart homepage</a>.</h4>



## Updates
- 10 January 2025: Experiment instructions have been released!
- 9 January 2025: The project description is now available. The code and experiment instructions will follow soon (slightly later than 10 January 2025).



## Highlights
- **User-Friendly:** Simple implementation with comprehensive experiment instructions.
- **Superior performance:** This implementation of the IconMatting model shows only a slight discrepancy compared to the results in the paper.
- **Expanded Test Set:** A newly released, manually created Image Matting test set.

## Requirements
Our experiments were conducted on an RTX 3090. The minimum required configuration is 16GB of VRAM. For optimal performance, a configuration with 24GB of VRAM or more is recommended.

## Environment
We have included our environment configuration and third-party library dependencies in the `env.yml` file. To ensure the project runs smoothly, please follow the steps below to set up the environment:

```bash
conda env create -f env.yml
    
## Prepare Your Data
1. Please contact Brian Price (bprice@adobe.com) requesting for the Adobe Image Matting dataset;
2. Composite the dataset using provided foreground images, alpha mattes, and background images from the COCO and Pascal VOC datasets. I slightly modified the provided `compositon_code.py` to improve the efficiency, included in the `scripts` folder. Note that, since the image resolution is quite high, the dataset will be over 100 GB after composition.
3. The final path structure used in my code looks like this:

````
$PATH_TO_DATASET/Combined_Dataset
├──── Training_set
│    ├──── alpha (431 images)
│    ├──── fg (431 images)
│    └──── merged (43100 images)
├──── Test_set
│    ├──── alpha (50 images)
│    ├──── fg (50 images)
│    ├──── merged (1000 images)
│    └──── trimaps (1000 images)
````

## Inference
Run the following command to do inference of IndexNet Matting/Deep Matting on the Adobe Image Matting dataset:

    python scripts/demo_indexnet_matting.py
    
    python scripts/demo_deep_matting.py
    
Please note that:
- `DATA_DIR` should be modified to your dataset directory;
- Images used in Deep Matting has been downsampled by 1/2 to enable the GPU inference. To reproduce the full-resolution results, the inference can be executed on CPU, which takes about 2 days.

Here is the results of IndexNet Matting and our reproduced results of Deep Matting on the Adobe Image Dataset:

| Methods | Remark | #Param. | GFLOPs | SAD | MSE | Grad | Conn | Model |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Deep Matting | Paper | -- | -- | 54.6 | 0.017 | 36.7 | 55.3 | -- |
| Deep Matting | Re-implementation | 130.55M | 32.34 | 55.8 | 0.018 | 34.6 | 56.8 | [Google Drive (522MB)](https://drive.google.com/open?id=1Uws86AGkFqV2S7XkNuR8dz5SOttxh7AY) |
| IndexNet Matting | Ours | 8.15M | 6.30 | 45.8 | 0.013 | 25.9 | 43.7 | Included |

* The original paper reported that there were 491 images, but the released dataset only includes 431 images. Among missing images, 38 of them were said double counted, and the other 24 of them were not released. As a result, we at least use 4.87% fewer training data than the original paper. Thus, the small differerce in performance should be normal.
* The evaluation code (Matlab code implemented by the Deep Image Matting's author) placed in the ``./evaluation_code`` folder is used to report the final performance for a fair comparion. We have also implemented a python version. The numerial difference is subtle.

## Training
Run the following command to train IndexNet Matting:

    sh train.sh
    
- `--data-dir` should be modified to your dataset directory.
- I was able to train the model on a single GTX 1080ti (12 GB). The training takes about 5 days. The current bottleneck appears to be the dataloader.

## Citation
If you find this work or code useful for your research, please cite:
```
@inproceedings{hao2019indexnet,
  title={Indices Matter: Learning to Index for Deep Image Matting},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}

@article{hao2020indexnet,
  title={Index Networks},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020}
}
```

## Permission and Disclaimer
This code is only for non-commercial purposes. As covered by the ADOBE IMAGE DATASET LICENSE AGREEMENT, the trained models included in this repository can only be used/distributed for non-commercial purposes. Anyone who violates this rule will be at his/her own risk.
