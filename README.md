Image Obfuscation for Privacy-Preserving Machine Learning
====

# 1 - File Structure

Below you can find the file structure of the repository with an explaination of what each folder was used for.

 - essai2/ : First tries for results on various models and datasets
 - clean/ : A first clean-up of essai2/ which was promptly abandonned
 - exps/ : The main folder in which most of the work was performed
   - results/ : The dumps of the outputs of most experiments used in the report
   - pdf_files/ : The pdf files of the figures used in the report
   - results_\*.ipynb : The notebooks that were used to generate tables and plots for the report
   - \*.ipynb : The other notebooks were used to perform the experiments, and to test various methods later inserted in \*.py files in final/.
 - final/ : The folder for the formatted code destined to external review
 - README.md : This file, the README

 
# 2 - Contents of final/

Here is a description of the files and folders in final/ destined to external review

 - results/ : contains the dumps for the results that were generated in exps/.
 - fixed-mixup.ipynb : to perform experiments for Section 3 and 4
 - pixel-shuffling.ipynb : to perform experiments for Section 5
 - pixel-grafting.ipynb : to perform experiments for Section 6
 - results_\*.ipynb : see 3 - Experiment results below
 - exp_models.py : code for the models used for the experiments: LeNet-like CNN and ResNet18
 - lambdas.py : code to generate various lambdas described in the report
 - pixelshuffling.py : code to provide functions and the PixelShuffler class for Pixel-Shuffling
 - results_deser.py : see 3 - Experiment results below
 - 

# 3 - Experiment results

In final/results you can find results files, almost all of which have been generated over the whole semester in exps/results and copied over in final/results. Some files are a little bit corrupted, a lot are placeholders or empty, and a substential part of them are never used in the project/report.

There are code files and notebooks provided that extract the experiment results from those files and generate tables and plots. Those notebooks are named results_\*.ipynb in reference to the category of results they read.

The code used to deserialize dumped outputs is found in results_deser.py

Here is the summary of the result notebooks and their use within the report:

final/results_benchmark.ipynb -- Section 3, "Benchmark"
final/results_initial.ipynb -- Subsection 4.4, "Initial results"
final/results_finding_sweetspot.ipynb -- Subsection 4.5, "Generalized Fixed-mixup"
final/results_sweetspot_vs_maxobf.ipynb -- Subsection 4.7, "Relationship between mean test accuracy and N"
final/results_pixel_shuffling.ipynb -- Section 5, "Exploring Pixel-Shuffling"
final/results_pixel_grafting.ipynb -- Section 6, "Exploring Pixel-Grafting"

