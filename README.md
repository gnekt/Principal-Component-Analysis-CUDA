# PCA in CUDA

By Giacomo Nunziati ([@nunziati](https://github.com/nunziati)) and Christian Di Maio ([@gnekt](https://github.com/gnekt))

A from-scratch implementation of **Principal Component Analysis** in three flavours: pure CPU, and two CUDA GPU approaches with different eigenvalue/eigenvector strategies.

The full report is available [here](https://drive.google.com/file/d/13pUux-1u03KkTQMhSiXNDL2ghPydaUmr/view?usp=sharing).

---

## Algorithm

All three versions follow the same pipeline:

1. **Normalize** the input matrix column-wise (zero mean, unit max)
2. **Covariance matrix** via `Normalized^T * Normalized`
3. **Eigenvalue decomposition** using the Jacobi iterative method
4. **Eigenvector computation** — this is where the versions differ
5. **Component selection** — keep the fewest principal components that capture 99.9% of variance

### How the versions differ

| | CPU | GPU v1 | GPU v2 |
|---|---|---|---|
| Eigenvalues | Serial Jacobi | Parallel Jacobi (find max + rotation kernels) | Parallel Jacobi (same) |
| Eigenvectors | Inverse power iteration (Moore-Penrose pseudo-inverse via Cholesky) | Same as CPU, but GPU-accelerated kernels | Computed **simultaneously** with eigenvalues — the Jacobi rotations are accumulated into an eigenvector matrix, so no separate step is needed |
| Steps timed | 4 (normalize, covariance, eigenvalues, eigenvectors) | 4 | 3 (normalize, covariance, eigenvalues+eigenvectors) |

---

## Building

**CPU version** — just g++:
```bash
cd cpu_ver
g++ -o cpu.main cpu.main.cpp
g++ -o cpu.experiment cpu.experiment.cpp
```

**GPU versions** — requires CUDA Toolkit (default path: `/usr/local/cuda-11.2`):
```bash
cd gpu_ver_1   # or gpu_ver_2
make
```

Override the toolkit path if needed: `make CUDA_PATH=/usr/local/cuda-12.0`

---

## Usage

All executables take the same arguments:

```
./<executable> <csv_file> <separator> <num_rows> <num_cols>
```

### Running PCA

```bash
# CPU
cd cpu_ver
./cpu.main "soil_data.csv" "," 3109 32

# GPU v1
cd gpu_ver_1
./gpu1.main "soil_data.csv" "," 3109 32

# GPU v2
cd gpu_ver_2
./gpu2.main "soil_data.csv" "," 3109 32
```

Output: `EigenVectors.csv` — the principal component eigenvectors ordered by importance, keeping only those that represent 99.9% of the information from the covariance matrix.

### Running experiments

The experiment programs run PCA across 11 different configurations (varying row/column counts across two datasets) and measure execution time for each step.

```bash
./cpu.experiment "kronos.cpu"
./gpu1.experiment "kronos.gpu1"
./gpu2.experiment "kronos.gpu2"
```

Each produces:
- One eigenvector CSV per experiment (`<name>.1.csv`, `<name>.2.csv`, ...)
- A timing measurements CSV (`<name>.measurement.csv`)

The measurement CSV format (each row = one experiment, values in seconds):
```
normalization_mean, normalization_std, normalization_min, normalization_max,
covariance_mean, covariance_std, covariance_min, covariance_max,
eigenvalues_mean, eigenvalues_std, eigenvalues_min, eigenvalues_max,
[eigenvectors_mean, eigenvectors_std, eigenvectors_min, eigenvectors_max]  # CPU & GPU1 only
```

> **Note:** The experiment programs require `soil_data.csv` (and optionally `hapt_train_set.csv`) to be in the same directory as the executable.

---

## Dataset

We used the [US Drought Meteorological Data](https://www.kaggle.com/cdminix/us-drought-meteorological-data) from Kaggle — `soil_data.csv` contains 3109 rows and 32 features of soil meteorological measurements.

---

## Results

Full computational time comparisons are available in [this spreadsheet](https://docs.google.com/spreadsheets/d/1VZl1sm0TSgjVY3uonsJZ0mZhHJEjjh3B_HvNUMFAsfU/edit?usp=sharing).
