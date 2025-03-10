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
  - [Graph-based approaches](#graph-based-approaches)
  - [Approaches accounting dependencies](#dependency-approaches)
  - [End-to-end approaches](#e2e-approaches)
  - [Instance-dependent model-based approaches](#instance-dependent-approaches)
  - [Approaches accounting fairness and bias](#fair-bias-approaches)
  - [Adversarial annotators](#adversarial-approaches)
  - [Reinforcement learning from Crowd feedback](#RLCF-approaches)
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
* Spectral methods meet EM: A provably optimal algorithm for crowdsourcing, Journal of Machine Learning Research, 2016, [[Paper]](https://jmlr.org/papers/volume17/14-511/14-511.pdf) [[Code]](https://github.com/zhangyuc/SpectralMethodsMeetEM)
* Blind multiclass ensemble classification, IEEE Trans. Signal Process., 2018. [[Paper]](https://ieeexplore.ieee.org/abstract/document/8421667) 
* Crowdsourcing via Pairwise Co-occurrences: Identifiability and Algorithms, NeurIPS, 2019. [[Paper]](https://openreview.net/pdf?id=HJl034rgIB) [[Code]](https://github.com/shahana-ibrahim/crowdsourcing)
* Crowdsourcing via annotator co-occurrence imputation and provable symmetric nonnegative matrix factorization, ICML, 2021. [[Paper]](https://proceedings.mlr.press/v139/ibrahim21a/ibrahim21a.pdf) [[Code]](https://github.com/shahana-ibrahim/crowdsourcing-via-co-occurrence-imputation)

