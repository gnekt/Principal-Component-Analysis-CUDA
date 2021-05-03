# PCA project realized by Giacomo Nunziati(@GiacomoNunziati) and Christian Di Maio

## Result of the work and how we construct the algorithm

Here you can see or download the report that we produce for this project

https://drive.google.com/file/d/13pUux-1u03KkTQMhSiXNDL2ghPydaUmr/view?usp=sharing


## Version:

- CPU version: compile it with g++ 
- GPU1 version: compile through the MakeFile with the command make under the directory associated
- GPU2 version: compile through the MakeFile with the command make under the directory associated

---

## How to use it:

Given that the input file has a csv format and it is also pre-processed (no value different from numbers are admitted),
you can call the PCA procedure as following:

CPU : cpu1.main <dataset_csv> <column_separator> <number_of_rows_csv> <number_of_columns_csv> 
- Eg. 
```
    ./main "soil_data.csv" "," 3109 32
```
GPU1 : gpu1.main <dataset_csv> <column_separator> <number_of_rows_csv> <number_of_columns_csv> 
- Eg. 
```
    ./gpu1.main "soil_data.csv" "," 3109 32
```
GPU2 : gpu2.main <dataset_csv> <column_separator> <number_of_rows_csv> <number_of_columns_csv> 
- Eg. 
```
    ./gpu2.main "soil_data.csv" "," 3109 32
```
---

## How we do the experiments in a more efficient way?
We create 3 different programs that will do the job for us, if you are curious to see how it works the make file include also the compile for the experiment program, but it is more hardcoded and require that "soil_data.csv" are in the same directory of the executable.


CPU : cpu.experiment <name_of_the_machine_on_which_it_will_be_run> 
- Eg. 
```
    ./cpu1.experiment "kronos.cpu"
```

GPU1 : gpu1.main <name_of_the_machine_on_which_it_will_be_run> 
- Eg. 
```
    ./gpu1.experiment "kronos.gpu1"
```

GPU2 : gpu2.main <name_of_the_machine_on_which_it_will_be_run> 
- Eg. 
```
    ./gpu2.experiment "kronos.gpu2"
```

### It will create the csv eigenvector files and also a csv of measurement formatted in this way:

### CPU & GPU1 each row represent an experiment, each value represent elapsed time to do this task expressend in seconds
```
normalization_mean,normalization_std,normalization_min,normalization_max,covariance_mean,covariance_std,covariance_min,covariance_max,eigenvalues_mean,eigenvalues_std,eigenvalues_min,eigenvalues_max,eigenvectors_mean,eigenvectors_std,eigenvectors_min,eigenvectors_max
```
### GPU2 each row represent an experiment, each value represent elapsed time to do this task expressend in seconds
```
normalization_mean,normalization_std,normalization_min,normalization_max,covariance_mean,covariance_std,covariance_min,covariance_max,eigenvectors_mean,eigenvectors_std,eigenvectors_min,eigenvectors_max
```
---

## Curiosity 
-If you are very curious here you can see all the result in terms of computational time: https://docs.google.com/spreadsheets/d/1VZl1sm0TSgjVY3uonsJZ0mZhHJEjjh3B_HvNUMFAsfU/edit?usp=sharing

-Here it is the data set that we used 
https://www.kaggle.com/cdminix/us-drought-meteorological-data

