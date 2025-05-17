We use the following R and Python versions:

R version 4.2.2
Python version: 3.10.9

We note that all the R and python files are compatible to be run with the latest version of all the packages used. Simply doing install.packages ("package name") for R packages used and pip install "package name" for python packages will suffice. For completeness, we include the version numbers of various packages and libraries used.


R libraries:

randomForest: ‘4.7.1.1’
glmnet: ‘4.1.6’

Python packages:

numpy: 1.23.5
sys (standard library, version not applicable)
matplotlib: 3.7.0
seaborn: 0.12.2
sklearn: 1.2.1
cvxpy: 1.4.0
pandas: 1.5.3
scipy: 1.10.0
joblib: 1.1.1
pickle (standard library, version not applicable)
geopandas: 0.14.0
pynndescent: 0.5.10
pymde: 0.1.18
statsmodels: 0.13.5
TensorFlow version:  2.14.0
TensorFlow Datasets version:  4.9.3


Follow the guide here for compiling in cython: https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html?fbclid=IwAR0zDvNZZEYDOSZgJeFK8_HmDqFZXW5bxLS0pHnTB3wOLSYszkXGRe_Dauk



To generate Figures 1 and 5, run the files code/Make Figs 1, 5 part 1.ipynb  and code/Make Figs 1, 5 part 2.ipynb 

To generate figures 2, 3, and 4, run the following files in the order:

Figure 2
Dataset preprocessing -> code/processing-cifar_10-and-mnist.ipynb
Experiment -> code/cifar10_mnist_experiment.py
Plot -> code/Make Figs 2, 3, 4.ipynb

Figure 3
Dataset preprocessing -> code/processing-cifar_10-and-mnist.ipynb
Experiment -> code/cifar10_mnist_experiment.py
Plot -> code/Make Figs 2, 3, 4.ipynb

Figure 4
Dataset preprocessing -> code/processing_imagenet.py / processing-imagenet.ipynb
Experiment -> code/cifar10_imagenet_experiment.py
Plot -> code/Make Figs 2, 3, 4.ipynb

To generate figures 6, 7, 8, and 9, run the file code/Make Figs 6, 7, 8, 9.ipynb

To generate figure 10, run the following files:

code/Make Fig 10-real estate.ipynb
code/Make Fig 10-weather history.ipynb
code/Make Fig 10-wine quality.ipynb