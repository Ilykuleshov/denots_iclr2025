# Datasets
All datasets can be either downloaded or generated automatically, in the corresponding notebooks:
 - InsectSound and UWGL can be downloaded in the [`uae.ipynb`](uae.ipynb).
 - Pendulum (and its smaller version, pendulum-small) can be generated in the [`generate_pendulum.ipynb`](generate_pendulum.ipynb).
 - Pendulum angles dataset for forecasting is generated in [`pendulum_angles.ipynb`](pendulum_angles.ipynb).
 - Sine mix is generated in  [`sine_mix.ipynb`](sine_mix.ipynb).
 - Sine-2 is generated in  [`sine.ipynb`](sine.ipynb).
 - The Bump dataset is generated in  [`bump.ipynb`](bump.ipynb).
 - Sepsis is downloaded in [`download_sepsis.sh`](download_sepsis.sh), and preprocessed in [`preprocess_sepsis.ipynb`](preprocess_sepsis.ipynb).


Preprocessing is done using the corresponding Jupyter notebooks, to make data access easy.
We use [Polars](https://pola.rs/), so it finishes quickly.
 