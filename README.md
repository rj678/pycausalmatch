
# pycausalmatch

pycausalmatch is a Python library for causal inference integrated with the
process of selecting suitable control groups

(I plan to develop this as one would a causal inference project for Big Data with the
intent of eventually deploying pipelines)


#### Description

The functionality that has been implemented so far is essentially a Python translation of the
features available in the R library: https://github.com/klarsen1/MarketMatching (v.1.1.7 - as of Dec 2020),
which combines 2 packages: https://github.com/dafiti/causalimpact and https://github.com/DynamicTimeWarping/dtw-python

The DTW package is used for selection of most suitable control groups.

The R library has a detailed README.

The causal impact from this Python version matches the impact for the test market ('CPH') in the example
in the R library, as shown in the plots in the `starter_example` notebook.

This is still an **alpha release** - I'm in the process of adding more features, and fixing
all the bugs soon!

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pycausalmatch.

```bash
pip install pycausalmatch
```

## Usage

```python
from pycausalmatch import R_MarketMatching as rmm

rmm.best_matches(**kwargs) # returns
rmm.inference(**kwargs) # returns

```

This package has only been tested for **a single test market** (I will test it for multiple test markets soon)


## Example Use case

I've added an example on the causal impact of Prop 99 in California in the notebook `prop_99_example`
under the notebooks/examples folder. I will keep updating this example as I develop the library further.




## TODOs

- [ ] Improve README!

- [ ] Add more examples (Prop 99 - CA)

- [ ] add tests, logging ...

- [ ] add statistical inference

- [x] use software project structure template

- [ ] Integrate into an MLOps workflow

- [ ] Add parallel execution (I plan to use Bodo)

- [ ] Add Streamlit and Dash app

- [ ] switch to https://github.com/WillianFuks/tfcausalimpact

- [ ] add remaining functionality of the R package





## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


#### Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
