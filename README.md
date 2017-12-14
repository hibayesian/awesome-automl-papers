# Awesome-AutoML-Papers 

A curated list of automated machine learning papers, articles, tutorials, slides and projects.

# Introduction to AutoML
Machine learning (ML) has achieved considerable successes in recent years and an ever-growing number of disciplines rely on it. However, this success crucially relies on human machine learning experts to perform the following tasks:
+ Preprocess the data
+ Select appropriate features
+ Select an appropriate model family
+ Optimize model hyperparameters
+ Postprocess machine learning models
+ Critically analyze the results obtained.

As the complexity of these tasks is often beyond non-ML-experts, the rapid growth of machine learning applications has created a demand for off-the-shelf machine learning methods that can be used easily and without expert knowledge. We call the resulting research area that targets progressive automation of machine learning *AutoML*.

AutoML draws on many disciplines of machine learning, prominently including
+ Bayesian optimization
+ Regression models for structured data and big data
+ Meta learning
+ Transfer learning, and
+ Combinatorial optimization.

# Table of Contents

+ [Papers](#papers)
  - [Bayesian Optimization](#bayesian-optimization)
  - [Local Search](#local-search)
  - [Meta Learning](#meta-learning)
  - [Particle Swarm Optimization](#particle-swarm-optimization)
  - [Random Search](#random-search)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Transfer Learning](#transfer-learning)
+ [Tutorials](#tutorials)
  - [Bayesian Optimization](#bayesian-optimization)
  - [Meta Learning](#meta-learning)
+ [Articles](#articles)
  - [Bayesian Optimization](#bayesian-optimization)
  - [Meta Learning](#meta-learning)
+ [Slides](#slides)
  - [Bayesian Optimization](#bayesian-optimization)
+ [Books](#books)
  - [Meta Learning](#meta-learning)
+ [Projects](#projects)
+ [Prominent Researchers](#prominent-researchers)
+ [Competitions](#competitions)

# Papers
### Bayesian Optimization
+ 2017 | Google Vizier: A Service for Black-Box Optimization | [`PDF`](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/46180.pdf)
+ 2015 | Efficient and Robust Automated Machine Learning | [`PDF`](https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf)
+ 2013 | Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms | [`PDF`](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/papers/autoweka.pdf)
+ 2012 | Practical Bayesian Optimization of Machine Learning Algorithms | [`PDF`](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
+ 2011 | Sequential Model-Based Optimization for General Algorithm Configuration(extended version) | [`PDF`](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf)
### Local Search
+ 2009 | ParamILS: An Automatic Algorithm Configuration Framework | Frank Hutter, et al. | JAIR | [`PDF`](https://arxiv.org/pdf/1401.3492.pdf)
### Meta Learning
+ 2008 | Cross-Disciplinary Perspectives on Meta-Learning for Algorithm Selection | [`PDF`](https://dl.acm.org/citation.cfm?id=1456656)
### Particle Swarm Optimization
+ 2017 | Particle swarm optimization for hyper-parameter selection in deep neural networks | Pablo Ribalta Lorenzo, et al. | GECCO | [`PDF`](https://dl.acm.org/citation.cfm?id=3071208)
+ 2008 | Particle swarm optimization for parameter determination and feature selection of support vector machines | Shih-Wei Lin, et al. | Expert Systems with Applications | [`PDF`](http://www.sciencedirect.com/science/article/pii/S0957417407003752)
### Random Search
+ 2016 | Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization | Lisha Li, et al. | arXiv |  [`PDF`](https://arxiv.org/pdf/1603.06560.pdf)
+ 2012 | Random Search for Hyper-Parameter Optimization | James Bergstra, Yoshua Bengio | JMLR | [`PDF`](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
### Reinforcement Learning
+ 2010 | Feature Selection as a One-Player Game | Romaric Gaudel, Michele Sebag | ICML | [`PDF`](https://hal.archives-ouvertes.fr/inria-00484049/document)
### Transfer Learning
+ 2016 | Flexible Transfer Learning Framework for Bayesian Optimisation | Tinu Theckel Joy, et al. | PAKDD | [`PDF`](https://link.springer.com/chapter/10.1007/978-3-319-31753-3_9)

# Tutorials
### Bayesian Optimization
+ 2010 | A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning | [`PDF`](https://arxiv.org/pdf/1012.2599v1.pdf)
### Meta Learning
+ 2008 | Metalearning - A Tutorial | [`PDF`](https://pdfs.semanticscholar.org/5794/1a4891f673cadf06fba02419372aad85c3bb.pdf)

# Articles
### Bayesian Optimization
+ 2016 | Bayesian Optimization for Hyperparameter Tuning | [`Link`](https://arimo.com/data-science/2016/bayesian-optimization-hyperparameter-tuning/)
### Meta Learning
+ 2017 | Why Meta-learning is Crucial for Further Advances of Artificial Intelligence? | [`Link`](https://chatbotslife.com/why-meta-learning-is-crucial-for-further-advances-of-artificial-intelligence-c2df55959adf)
+ 2017 | Learning to learn | [`Link`](http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)

# Slides
### Bayesian Optimization
+ Bayesian Optimisation | [`PDF`](https://github.com/hibayesian/awesome-automl-papers/blob/master/resource/slides/%5Bslides%5D-bayesian-optimisation.pdf)
+ A Tutorial on Bayesian Optimization for Machine Learning | [`PDF`](https://github.com/hibayesian/awesome-automl-papers/blob/master/resource/slides/%5Bslides%5D-a-tutorial-on-bayesian-optimization-for-machine-learning.pdf)

# Books
### Meta Learning
+ 2009 | Metalearning - Applications to Data Mining | Springer | [`PDF`](http://www.springer.com/la/book/9783540732624)

# Projects
+ Advisor | `Python` | `Open Source` | [`Code`](https://github.com/tobegit3hub/advisor)
+ auto-sklearn | `Python` | `Open Source` | [`Code`](https://github.com/automl/auto-sklearn)
+ Auto-WEKA | `Java` | `Open Source` | [`Code`](https://github.com/automl/autoweka)
+ Hyperopt | `Python` | `Open Source` | [`Code`](https://github.com/hyperopt/hyperopt)
+ Hyperopt-sklearn | `Python` | `Open Source` | [`Code`](https://github.com/hyperopt/hyperopt-sklearn)
+ SigOpt | `Python` | `Commercial` | [`Link`](https://sigopt.com/)
+ SMAC3 | `Python` | `Open Source` | [`Code`](https://github.com/automl/SMAC3)
+ RoBO | `Python` | `Open Source` | [`Code`](https://github.com/automl/RoBO)
+ BayesianOptimization | `Python` | `Open Source` | [`Code`](https://github.com/fmfn/BayesianOptimization)
+ Scikit-Optimize | `Python` | `Open Source` | [`Code`](https://github.com/scikit-optimize/scikit-optimize)
+ HyperBand | `Python` | `Open Source` | [`Code`](https://github.com/zygmuntz/hyperband)
+ BayesOpt | `C++` | `Open Source` | [`Code`](https://github.com/rmcantin/bayesopt)
+ Optunity | `Python` | `Open Source` | [`Code`](https://github.com/claesenm/optunity)
+ TPOT | `Python` | `Open Source` | [`Code`](https://github.com/rhiever/tpot)

# Prominent Researchers
+ [Frank Hutter](http://aad.informatik.uni-freiburg.de/people/hutter/index.html)

# Competitions
+ AutoML2018 challenge | Nov 30, 2017 - March 15, 2018 | 4Paradigm, ChaLearn | [`Link`](https://competitions.codalab.org/competitions/17767)

# Licenses
Awesome-AutoML-Papers is available under Apache Licenses 2.0.

# Contact & Feedback
If you have any suggestions (missing papers, new papers, key researchers or typos), feel free to pull a request. Also you can mail to:
+ hibayesian (hibayesian@gmail.com).
