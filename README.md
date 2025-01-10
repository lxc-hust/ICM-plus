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
- **10 January 2025:** Experiment instructions have been released!
- **9 January 2025:** The project description is now available. The code and experiment instructions will follow soon (slightly later than 10 January 2025).



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
```

## Prepare model checkpoints
1. The IconMatting model checkpoints are open-source by <a href="https://pan.baidu.com/share/init?surl=HPbRRE5ZtPRpOSocm9qOmA&pwd=BA1c">HUST TinySmart</a>.
````bash
conda env create -f env.yml
````
2. <a href="https://huggingface.co/stabilityai/stable-diffusion-2-1">Stable Diffusion v2-1</a> is also required. To download it, run the following commands:
````
huggingface-cli download --resume-download stabilityai/stable-diffusion-2-1 --local-dir your/local/dir
````
Make sure that the `your/local/dir` directory matches the path set in `config/eval.yaml`.


## Prepare test datasets
1.<a href="https://pan.baidu.com/share/init?surl=ZJU_XHEVhIaVzGFPK_XCRg&pwd=BA1c">ICM-57</a> dataset is open-source by HUST-TinySmart.
2.ICM-plus dataset has been submitted.
It is recommended that your dataset be organized with the following structure:
````
$./datasets
├──── ICM57
│    ├──── image
│    ├──── alpha
│    └──── trimap
├──── your_test_set
│    ├──── image
│    ├──── alpha
│    └──── trimap
````
If the dataset is missing trimap information, you can generate it by running:
````
python trimap_gen.py
````

## Evaluation
1.Make sure to update the file reading path in `./icm/data/image_file.py` to point to your dataset directory. Additionally, ensure that the `dataset_name` field in `./config/eval.yaml` matches the name of your dataset.
2.Run the following command to generate the predicted alpha matte:

````
python eval.py --checkpoint PATH_TO_MODEL --save_path results/ --config config/eval.yaml --seed your_seed
````

3.To calculate the four metrics: MSE, SAD, Grad, and Conn, run the following command:

````
python calculate.py --pred_folder ./results --gt_folder your/dataset/alpha/matte
````

