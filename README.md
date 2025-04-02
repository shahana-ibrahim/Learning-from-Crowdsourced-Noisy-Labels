# Learning-from-Crowdsourced-Noisy-Labels
A curated list of papers and their codes in the domain of learning from crowdsourced noisy labels.
More detailed survey and in-depth details of many of these techniques can be found in our tutorial paper
[Learning From Crowdsourced Noisy Labels: A Signal Processing Perspective](https://arxiv.org/abs/2407.06902).

## Content
  - [Weighted majority voting](#weighted-majority-voting)
  - [Expectation maximization approaches](#expectation-maximization-approaches)
  - [Spectral methods](#spectral-methods)
  - [Moment-based approaches](#moment-based-approaches)
  - [Optimization-based approaches](#optimization-based-approaches)
  - [Approaches accounting dependencies](#approaches-accounting-dependencies)
  - [End-to-end approaches](#end-to-end-approaches)
  - [Instance-dependent model-based approaches](#instance-dependent-model-based-approaches)
  - [Adversarial annotators](#adversarial-annotators)
  - [Reinforcement learning from Crowd feedback](#reinforcement-learning-from-crowd-feedback)
  - [Benchmarking datasets](#benchmarking-datasets)
---

## Weighted Majority Voting 
* The weighted majority algorithm. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0890540184710091)
* Exact Exponent in Optimal Rates for Crowdsourcing. [[Paper]](https://proceedings.mlr.press/v48/gaoa16.pdf)

## Expectation Maximization Approaches
* Maximum likelihood estimation of observer error-rates using the EM algorithm, Applied Statistics, 1979. [[Paper]](https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2346806); Unofficial python implementation [[Code]](https://github.com/dallascard/dawid_skene)
* Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise, NeurIPS, 2009. [[Paper]](https://papers.nips.cc/paper_files/paper/2009/hash/f899139df5e1059396431415e770c6dd-Abstract.html); Unofficial python implementation [[Code]](https://github.com/notani/python-glad)
* Learning from crowds, Journal of Machine Learning Research, 2010. [[Paper]](https://jmlr.csail.mit.edu/papers/v11/raykar10a.html); Unofficial python implementation [[Code]](https://github.com/fmenat/PyLearningCrowds)

## Spectral Methods
* Who moderates the moderators? Crowdsourcing abuse detection in user-generated content, Proceedings of the ACM Conference on Electronic Commerce, 2011. [[Paper]](https://dl.acm.org/doi/10.1145/1993574.1993599)
* Aggregating crowdsourced binary ratings, Proceedings of the International Conference on World Wide Web, 2013. [[Paper]](https://dl.acm.org/doi/10.1145/2488388.2488414)
* Estimating the accuracies of multiple classifiers without labeled data, Artificial Intelligence and Statistics, 2015. [[Paper]](https://proceedings.mlr.press/v38/jaffe15.pdf)


## Moment-based Approaches
* Spectral methods meet EM: A provably optimal algorithm for crowdsourcing, Journal of Machine Learning Research, 2016, [[Paper]](https://jmlr.org/papers/volume17/14-511/14-511.pdf) [[Code]](https://github.com/zhangyuc/SpectralMethodsMeetEM)
* Blind multiclass ensemble classification, IEEE Trans. Signal Process., 2018. [[Paper]](https://ieeexplore.ieee.org/abstract/document/8421667) 
* Crowdsourcing via Pairwise Co-occurrences: Identifiability and Algorithms, NeurIPS, 2019. [[Paper]](https://openreview.net/pdf?id=HJl034rgIB) [[Code]](https://github.com/shahana-ibrahim/crowdsourcing)
* Crowdsourcing via annotator co-occurrence imputation and provable symmetric nonnegative matrix factorization, ICML, 2021. [[Paper]](https://proceedings.mlr.press/v139/ibrahim21a/ibrahim21a.pdf) [[Code]](https://github.com/shahana-ibrahim/crowdsourcing-via-co-occurrence-imputation)

## Optimization-based Approaches
* Resolving conflicts in heterogeneous data by truth discovery and source reliability estimation, ACM SIGMOD, 2024. [[Paper]](https://dl.acm.org/doi/10.1145/2588555.2610509)
* Learning from the Wisdom of Crowds by Minimax Entropy, NeurIPS, 2012. [[Paper]](https://papers.nips.cc/paper_files/paper/2012/hash/46489c17893dfdcf028883202cefd6d1-Abstract.html)
* Max-margin majority voting for learning from crowds, NeurIPS, 2015. [[Paper]](https://papers.nips.cc/paper_files/paper/2015/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html)
* Budget-optimal task allocation for reliable crowdsourcing systems, Operations Research, 2014. [[Paper]](https://pubsonline.informs.org/doi/abs/10.1287/opre.2013.1235?journalCode=opre)
* Learning From Crowds With Multiple Noisy Label Distribution Propagation, IEEE Transactions on Neural Networks and Learning Systems, 2022. [[Paper]](https://ieeexplore.ieee.org/document/9444560)

## Approaches accounting dependencies
* Unsupervised ensemble classification with sequential and networked data, IEEE Transactions on Knowledge and Data Engineering, 2022. [[Paper]](https://ieeexplore.ieee.org/document/9302602)
* A Bayesian Approach for Sequence Tagging with Crowds, Proceedings of the Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing, 2019. [[Paper]](https://aclanthology.org/D19-1101/)
* Unsupervised ensemble learning with dependent classifiers, AISTATS, 2016. [[Paper]](https://proceedings.mlr.press/v51/jaffe16.html)
* Identifying Dependent Annotators in Crowdsourcing, Asilomar, 2022. [[Paper]](https://ieeexplore.ieee.org/document/10052052)
* Bayesian classifier combination, AISTATS, 2012. [[Paper]](https://proceedings.mlr.press/v22/kim12.html)


## End-to-end Approaches
* Deep Learning from Crowds, AAAI Conference on Artificial Intelligence, 2018. [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/f86890095c957e9b949d11d15f0d0cd5-Abstract.html) [[Code]](https://github.com/fmpr/CrowdLayer)
* Deep Learning From Crowdsourced Labels: Coupled Cross-Entropy Minimization, Identifiability, and Regularization, ICLR, 2023. [[Paper]](https://openreview.net/forum?id=_qVhsWyWB9) [[Code]](https://github.com/shahana-ibrahim/end-to-end-crowdsourcing)
* Learning From Noisy Labels by Regularized Estimation of Annotator Confusion, CVPR, 2019. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tanno_Learning_From_Noisy_Labels_by_Regularized_Estimation_of_Annotator_Confusion_CVPR_2019_paper.pdf)
* Max-MIG: An Information Theoretic Approach for Joint Learning from Crowds, ICLR, 2019. [[Paper]](https://openreview.net/forum?id=BJg9DoR9t7) [[Code]](https://github.com/Newbeeer/Max-MIG)
* Structured probabilistic end-to-end learning from crowds, International Joint Conference on Artificial Intelligence, 2020. [[Paper]](https://www.ijcai.org/proceedings/2020/210)
* Deep learning from multiple noisy annotators as a union, IEEE Transactions on Neural Networks and Learning Systems, 2023. [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16730/16537)
* Learning from crowds by modeling common confusions, AAAI Conference on Artificial Intelligence, 2021. [[Paper]](https://ieeexplore.ieee.org/document/9765651)  

## Instance-dependent Model-based Approaches
* Disentangling Human Error from Ground Truth in Segmentation of Medical Images, NeurIPS, 2020. [[Paper]](https://proceedings.neurips.cc/paper/2020/file/b5d17ed2b502da15aa727af0d51508d6-Paper.pdf) [[Code]](https://github.com/moucheng2017/Med-Noisy-Labels)
* Label correction of crowdsourced noisy annotations with an instance-dependent noise transition model, NeurIPS, 2023. [[Paper]](https://openreview.net/forum?id=nFEQNYsjQO)
* Noisy Label Learning with Instance-Dependent Outliers: Identifiability via Crowd Wisdom, NeurIPS, 2024. [[Paper]](https://openreview.net/pdf?id=HTLJptF7qM) [[Code]](https://github.com/ductri/COINNet)

## Adversarial annotators
* Adversarial Crowdsourcing Through Robust Rank-One Matrix Completion, NeurIPS, 2020. [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/f86890095c957e9b949d11d15f0d0cd5-Abstract.html)
* Detecting adversaries in Crowdsourcing, IEEE International Conference on Data Mining, 2021. [[Paper]](https://ieeexplore.ieee.org/document/9678998/)

## Reinforcement learning from Crowd feedback
* Crowd-PrefRL: Preference-based reward learning from crowds, arXiv preprint, 2024. [[Paper]](https://arxiv.org/abs/2401.10941)
* Aligning crowd feedback via distributional preference reward modeling, arXiv preprint, 2024. [[Paper]](https://arxiv.org/abs/2402.09764)
* MaxMin-RLHF: Alignment with diverse human preferences, arXiv preprint, 2024. [[Paper]](https://arxiv.org/abs/2402.08925)

## Benchmarking Datasets
* [LabelMe](http://fprodrigues.com//deep_LabelMe.tar.gz)
* [MovieReviews](http://fprodrigues.com//deep_MovieReviews.tar.gz)
* [CONLL NER](http://fprodrigues.com//deep_ner-mturk.tar.gz)
* [CIFAR10n](http://noisylabels.com/)
* [CIFAR100n](http://noisylabels.com/)
* [ImageNet15n](https://github.com/ductri/COINNet)
* [Music](https://github.com/shahana-ibrahim/end-to-end-crowdsourcing/tree/master/data/Music)
