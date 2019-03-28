### Fully Automatic Morphing

The goal of this work is to generate, end to end, an image morphing between two images.
In order to obtain image morphing, a warp field between two images is required. The field generation process is based on **Visual Attribute Transfer through Deep Image Analogy**, Liao et. al. [[paper]](https://arxiv.org/abs/1705.01088). <br />
The field was generated by applying coarse-to-fine patchMatch between feature maps and then optimized in order to accomplish smoothness. 

This work has been done with Navot Oz.

## RUN
1. Update the .json file with your preferences. notice you can copy the opt.json file and update it as you wish! 
2. run:
```bash
python3 main.py --opt opt.json
```
3. The results will appear in 'result_path/name'
4. running the same opt file without chaning the name of the results dir or moving it will add a timestamp to the name of the results dir.
5. The opt files in the supplementry are backward-compatible to the program.
6. in order to continue training a previous session, remove the '//' from the 'load' section and set the "dir" to the one containing the .pickle files of the features. Change the "level" to the level that was trained last.
