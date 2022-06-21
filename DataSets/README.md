## [`/6M-SZZ-2020/`](https://github.com/sticeran/NoiseDetection/tree/main/DataSets/6M-SZZ-2020/)
The original 6M-SZZ-2020 data set is a noisy data set (i.e., defect labels as labels containing noise) open-sourced by the literature [1]. We selected 248 versions of it that contain noisy labels for our experiments.

## [`/IND-JLMIV+R-2020/`](https://github.com/sticeran/NoiseDetection/tree/main/DataSets/IND-JLMIV+R-2020/)
The original IND-JLMIV+R-2020 data set is a clean data set (i.e., defect labels as the ground truth) open-sourced by the literature [1]. On this basis, we filtered inconsistent labels to further reduce potentially noisy labels. For the instances filtered in the IND-JLMIV+R-2020 data set, we similarly filtered them in the 6M-SZZ-2020 data set to guarantee the consistency of the instances in both data sets.

## [`/(groundTruth)mislabels/`](https://github.com/sticeran/NoiseDetection/tree/main/DataSets/(groundTruth)mislabels/)
The defect labels from the IND-JLMIV+R-2020 data set after filtering inconsistent labels are used as the ground truth for the real noisy labels identified in the 6M-SZZ-2020 data set. This is used as the real noise data for model validation.

## References
[1]	S. Herbold, A. Trautsch, F. Trautsch. Issues with SZZ: an empirical assessment of the state of practice of defect prediction data collection. arXiv preprint arXiv:1911.08938v2, 2020.