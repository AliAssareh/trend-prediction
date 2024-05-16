# Detecting Trend of Cryptocurrency Rate of Exchange Using Multiple Series

This repository contains the code and results for my Master's thesis titled "Detecting Trend of Cryptocurrency Rate of Exchange Using Multiple Series." The study aims to predict the Bitcoin U.S. dollar exchange rate trend by defining a suitable classification problem and exploring various methodologies for feature extraction and model training.

## Repository Structure

- **Data**: Contains processed data files.
- **Documentation**: Includes detailed instructions on how to run the notebooks and additional documentation.
- **ModelingUtils**: Contains the code for all models used in the study.
- **preprocessing.ipynb**: Prepares the data for training and testing the models.
- **btc_models.ipynb**: Implements the best-performing model which encodes input timeframes and predicts market trends with SVM.
- **mix_models.ipynb**: Implements models that use both endogenous and exogenous data.
- **WD_vs_CWD.ipynb**: Compares the performance of the Wide-Deep network versus the Convolutional Wide-Deep network.
- **stationary_vs_nonstationary.ipynb**: Compares the performance of the model with stationary data versus non-stationary data.

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Run `preprocessing.ipynb` to prepare the data for model training and testing. This notebook saves the processed data into the Data folder.
2. **Model Comparison**: Explore the various model comparison notebooks:
    - `btc_models.ipynb`
    - `mix_models.ipynb`
    - `WD_vs_CWD.ipynb`
    - `stationary_vs_nonstationary.ipynb`
3. **Running Models**: Follow the instructions in the Documentation folder to run the notebooks and reproduce the results.


## Key Innovations

This study introduces several innovations:
- **Multiple Timeframes**: Utilizes 4h, 1h, and 15m timeframes.
- **Stationary Data**: Proposes a new method to make chart data and indicators stationary.
- **Dimensionality Reduction**: Introduces a new feature extraction method called Variational Nonlinear Neighborhood Component Analysis.
- **Target Definition**: Develops a new method for target definition in volatile cryptocurrency markets to avoid market noise.

## License

This project is licensed under the GNU General Public License. See the [LICENSE](LICENSE) file for details.


