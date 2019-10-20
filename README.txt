EM algorithm with Guassian
--------------------------
Guo Li 
NetID: gli27 
CSC 446
Apr 1 2019 
--------------------------


Briefly Description
-------------------
This project is a simple implementation of Expectation Maximum algorithm with Gaussian Mixture. The basic idea of this algorithm is to find an appropriate way to cluser those data point, in another words, this is a typical unsupervised machine learning problem. As what we learn from k-means algorithm, we design two steps to approach to the ideal solution, expectation and maximization steps. For expectation step, we need to calculate the expectation for each cluster. And as for maximization step, we need to use the expectation to calculate the parameters for current model. The general evaluation method we use in k-means is Euclidean distance. But this cannot be done for many complex circumstances. Then this mixture of gaussian come to people's attention and we can use multiply gaussian model with different means and convariances together to fitting any complex data distribution. We basically no longer calculate the distance but the probability of each data point belonging to that cluster. Then the expectation step of EM algorithm become calculate the probability of each point to every cluster. The maximization step become the update process for each cluster's parameters: lambda, mean and convariance. The reason why we cannot use stochastic gradient method is because we cannot solve the derivation of sum of log function. Then EM algorithm actually provide us a creative way to solve this kind of problems.


Extra args used in this assignment
----------------------------------
We actually only add one more parameter to the mutual group, --plot. When this parameter provided, the program will run and plot the analysis graph of iteration and cluster number's effect on thie problem.


Method of Initialize Model
--------------------------
For this project, we need to initialize the beginning parameters of the model in order to finish the first E step. Then we need a relatively appropriate way to initialize the model. For there, we initialize the lambdas with dirichlet distribution since all lambdas should be summed to 1. As for means(mus), we observe the data distribution and find it's around the origin dot. In this case, we take a standard normal distribution as a good way to initilize this parameters, we use 10 multiplies the result of random generated standard normal distribution values to generate the mean(mus). Last, we use the convariance the the original data set to initialize the convariance of each clusters. And for this part, since the means we given in advanced is relatively large, we also set relatively large convariance(3 multiplies the origin convariance), which will give a relatively large probability for data point far away from it and help us to train faster. 


Performance over hyperparameters (iterations, cluster_num, convariance_tied)
----------------------------------------------------------------------------
To clarify each parameters, here we give some breifly description for every of them.
Iteration is the time of training, which is the combination of E and M step. Since the basic idea of the algorithm is to fitting the data set, the training time determines the fitting conditions.
Cluster_num, is the cluster numbers user designate. This is the number of gaussian we mixed together to fit the data. This can also be considered as k in the k-means algorithm.
Convariance_tied, which means all gaussian models will use the same convariance matrix and have the same gaussian distribution shape.

As what you can see from the analysis graph (F1, F2 are models with individual convariance, F3, F4 are identical convariance), the average log likelihood is increasing in general with the increasing of iteration times and it will likely to converge at some points. For cluster numbers, generally it will make the train average likelihood increasing either but will cause overfitting problem, which will cause the dev likelihood decreasing at some points. As for convariance_tied parameter, we can find out that one single convariance will cause the average log likelihood decrease very quick at some point. This is because we can only change one single gaussian shape to fit the data, but data sets are always with unregular shape. So we can not always find a appropriate oval to fit an unregular data set distribution.
