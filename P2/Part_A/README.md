### part A) ### 

The code analyzing SGD on the Franke Function using OLS and Ridge regression can be found in the notebooks OLS_SGD_on_Franke_function.ipynb and Ridge_SGD_on_Franke_function.ipynb, respectively. The Class containing schedulers is taken from Morten Hjorth-Jensen's lecture notes for Week43 (https://github.com/CompPhysics/MachineLearning/blob/master/doc/LectureNotes/week43.ipynb), but comments and small adjustments are added to help us solve the tasks.
The function "designMatrix" is inspired by Morten Hjorth-Jensen's "create_X", also from the lecture notes of week 43. It is However adjusted to remove the intercept. 
The code makes use of the libraries SciKit learn and Autograd/JAX, as well as the common NumPy, Matplotlib and Pandas.
 
The code explores GD and SGD with different values of $\eta$, $\lambda_2$, batch size, $\eta$ tuning algorithm and number of epochs. Some plots are made for our own understanding and not included in the report. 
