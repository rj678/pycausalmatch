
# pycausalmatch

pycausalmatch is a Python library for causal inference integrated with the
process of selecting suitable control groups


#### Description

The functionality that has been implemented so far is essentially a Python translation of the
features available in the R library: https://github.com/klarsen1/MarketMatching , which combines 2 packages:
https://github.com/dafiti/causalimpact and https://github.com/DynamicTimeWarping/dtw-python

The DTW package is used for selection of most suitable control groups.

The R library has a detailed README.

The causal impact from this Python version matches the impact for the test market ('CPH') in the example
in the R library, as shown in the plots in the `starter_example` notebook.

This is still an **alpha release** - I will hopefully get around to adding more features, and fixing
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

## TODOs

- [ ] Improve README!

- [ ] add tests

- [ ] add statistical inference

- [ ] use software project structure template

- [ ] Integrate into an MLOps workflow

- [ ] Add parallel execution (I plan to use Bodo)

- [ ] Add Streamlit app

- [ ] switch to https://github.com/WillianFuks/tfcausalimpact

- [ ] add remaining functionality of the R package


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
