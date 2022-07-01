# The replication toolkit for the paper "Lightweight Noisy Labels Prediction Towards to Noise Distribution Characteristics"

## Titile: Lightweight Noisy Labels Prediction Towards to Noise Distribution Characteristics

Our work aim to analyze the characteristics of real noisy labels introduced by the currently popular label collection approach SZZ [1] in the multi-version defect data sets and to propose the targeted, simple and effective noise detection approach. Based on the unique characteristics of real noisy labels, we propose a lightweight noisy label prediction approach NEAT towards to the real distribution characteristics of noisy labels. The NEAT approach heuristically infer (predict) the noisy labels on the target version based on the noise knowledge on the adjacent historical version, according to the characteristic that the noise instances between adjacent versions are homogeneous.

## Quick Start

### (1) [`/DataSets/`](https://github.com/sticeran/NoiseDetection/tree/main/DataSets/) In this folder, the 6M-SZZ-2020 data set is the noisy data set (i.e., defect labels as labels containing noise) open-sourced from the literature [2].The IND-JLMIV+R-2020 data set is the clean data set (i.e., defect labels as the ground truth) open-sourced from the literature [2]. Please refer to the [`/DataSets/README.md`](https://github.com/sticeran/NoiseDetection/tree/main/DataSets/README.md) for details.

### (2) [`/Metric extractor/`](https://github.com/sticeran/NoiseDetection/tree/main/Metric%20extractor/) In this folder, the folder Metric extractor holds the feature collection scripts we implemented. Please refer to the [`/Metric extractor/README.md`](https://github.com/sticeran/SnoringNoise/tree/main/Metric%20extractor/README.md) for details.

### (3) [`/NoiseDetectionProgram/`](https://github.com/sticeran/NoiseDetection/tree/main/NoiseDetectionProgram/) This folder holds the experimental programs for reproducing the experimental results in RQ1, RQ2 and RQ3. Please refer to the [`/NoiseDetectionProgram/README.md`](https://github.com/sticeran/NoiseDetection/tree/main/NoiseDetectionProgram/README.md) for details.
 

If you use the data sets (data sets after filtering inconsistent labels on the original 6M-SZZ-2020 and IND-JLMIV+R-2020 multi-version defect data sets [2]) or the program code, please cite our paper "", thanks.

## References
[1] J. Sliwerski, T. Zimmermann, A. Zeller. When do changes induce fixes?. ACM SIGSOFT Software Engineering Notes, 30(4), 2005: 1-5.  
[2] S. Herbold, A. Trautsch, F. Trautsch. Issues with SZZ: an empirical assessment of the state of practice of defect prediction data collection. arXiv preprint arXiv:1911.08938v2, 2020.  
