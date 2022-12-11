# DehazeRL
Intelligent Video Dehazing using Reinforcement Learning

Objective

We present a reinforcement-learning controlled neural network for dehazing consecutive video frames that we name "DehazeRL." Prior work in dehazing focuses on dehazing images without analyzing spatio-temporally correlated images. We alter code published by \citet{Bajaj2021ReinforcementLO} originally intended for image dehazing and apply the modified code to dehaze video frames within the REVIDE dataset \citep{Li2019a}. Using this neural network, we achieve a PNSR of 20.898 and an SSIM of 0.721. 

Dataset
REVIDE Data folders can be found in: 
https://github.com/joshuaeh/DehazeRL/tree/master/REVIDE_inside

Requirements
pip install sewar
pip install stable-baselines3[extra]

Running the code
Link to end to end jupyter notebook
https://github.com/joshuaeh/DehazeRL/blob/master/PML_project_Video_Dehazing.ipynb
 
Sample Results:
<img width="926" alt="Screen Shot 2022-12-11 at 3 08 19 PM" src="https://user-images.githubusercontent.com/32895615/206928953-ed44d793-45de-4774-a915-e5cdd7286902.png">


