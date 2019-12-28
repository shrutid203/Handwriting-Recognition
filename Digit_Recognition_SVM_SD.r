#################################################
# SVM Assignment: Handwritten Digit Recognition #
#################################################

# 1. Business Understanding and Problem Statement:
# Identify hand written digits taken by a scanner, tablet or any digital device.
# The model should correctly identify digits based on pixels given as features

# 2. Installing and loading libraries: 
#install.packages("kernlab")
#install.packages("caret")
#install.packages("readr")
#install.packages("ggplot2")
#install.packages("gridExtra")

#Loading  libraries

library(kernlab)
library(readr)
library(caret)
library(ggplot2)
library(dplyr)
library(gridExtra)

#3. EDA and Data Preparation
#Setting Working Directory
setwd("C:/Users/shruti.diwakar/Desktop")

#3.1 Loading Data

train <-read.csv("mnist_train.csv", stringsAsFactors = F, header = F)
test<- read.csv("mnist_test.csv", stringsAsFactors = F, header = F)

#3.2 Checking Structure  of  datasets
str(train)
str(test)

#3.3 check contents of datasets
head(train,10)
head(test,10)
# We notice the datasets dont have appropriate column names. So we name them.
colnames(train)[1]<-"Digit"
colnames(test)[1]<-"Digit"



#3.4 summarising the datasets

summary(train)
summary(test)
# The data is clearly a 28x28 image since it has 784 features.

#3.5 check for missing values : No missing Values

sum(is.na(train))
sum(is.na(test))

#3.6 Checking for duplicate rows : No duplicate rows
sum(duplicated(train))
sum(duplicated(test))

#3.7 Convert Digit variable to a factor variable for the classifier
train$Digit<-factor(train$Digit)
test$Digit <-factor(test$Digit)

#3.8 Number of images available for each digit 
dig_summary<-train %>% group_by(Digit) %>% summarise(count_of_images=n())
dig_summary
# All digits have around 6000 images of handwrting each.


#3.9 Plot 28x28 MNIST dataset on a grid for visualization
# We will try to plot the data for digit "8"
# The image is a 28x28 matrix with integers between 0 to 784

digit_8<-head(train[train$Digit==8,],1)
vec_8<-as.vector(digit_8[,c(2:785)])
matrix_8<-matrix(unlist(vec_8),nrow=28,ncol=28,byrow = T)
# Defining function to Rotate the matrix
rotate <- function(x) t(apply(x, 2, rev)) 
# Plot the matix
image(rotate(matrix_8),col=grey.colors(255))
# As we can see from the image , it is the number 8.



#######################
#4. Modeling the data #
#######################

#4.1 Since Train data set is huge it will be computationally intensive. So we will sample the train dataset
# Taking 10% of the data 

set.seed(1)
sample_train = sample(1:nrow(train), 0.1*nrow(train))
train_data = train[sample_train, ]

#4.2 Using Linear Kernel for modelling
linear_mod <- ksvm(Digit~ ., data = train_data, scale = FALSE, kernel = "vanilladot")
pred_linear<- predict(linear_mod, test)

#4.2.1 confusion matrix - Linear Kernel
confusionMatrix(pred_linear,test$Digit)

#4.2.2 Overall Statistics

#Accuracy : 0.9139   
#95% CI : (0.9082, 0.9193)
#No Information Rate : 0.1135         
#P-Value [Acc > NIR] : < 2.2e-16      
# The accuracy is good for linear model. Let's try to use RBF kernel and see if it improves accuracy

#4.3 Using RBF Kernel
rbf_mod <- ksvm(Digit~ ., data = train_data, scale = FALSE, kernel = "rbfdot")
pred_rbf<- predict(rbf_mod, test)
# Cost C =1 and sigma = 1.6x 10^-7
# We will try to vary this hyperparameters in hyperparameter tuning.
#4.3.1 confusion matrix - RBF Kernel
confusionMatrix(pred_rbf,test$Digit)

#4.3.2 Overall Statistics

#Accuracy : 0.952        
#95% CI : (0.9476, 0.9561)
#No Information Rate : 0.1135          
#P-Value [Acc > NIR] : < 2.2e-16  

#The accuracy has increased! Hence RBF kernel is better suited for this dataset

##############################################
#5.Cross Validation and hyperparameter tuning#
##############################################


# We are going to do a grid search for the sigma and c hyperparameters.
# We are going to do a 3 cross validation
# We are going to pass these variables into train function

tr_control  <- trainControl(method="cv", number=3)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
# Taking parameters around 1.6 x 10^-7 since our initial parameters are around that, and it got good accuracy
set.seed(7)
grid <- expand.grid(.sigma=c(0.00000015,0.0000002,0.0000005), .C=c(0.5,1,1.5,2) )

fit.svm <- train(Digit~., data=train_data, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=tr_control)
print(fit.svm)
# Output:-
#Resampling: Cross-Validated (3 fold) 
#Summary of sample sizes: 3998, 4002, 4000 
#Resampling results across tuning parameters:
  
 #sigma    C    Accuracy   Kappa    
 # 1.5e-07  0.5  0.9325015  0.9249385
 # 1.5e-07  1.0  0.9418330  0.9353183
 # 1.5e-07  1.5  0.9460013  0.9399539
 # 1.5e-07  2.0  0.9483350  0.9425498
 # 2.0e-07  0.5  0.9393348  0.9325390
 # 2.0e-07  1.0  0.9471676  0.9412506
 # 2.0e-07  1.5  0.9516683  0.9462572
 # 2.0e-07  2.0  0.9526685  0.9473691
 # 5.0e-07  0.5  0.9475026  0.9416258
 # 5.0e-07  1.0  0.9568360  0.9520040
 # 5.0e-07  1.5  0.9578353  0.9531150
 # 5.0e-07  2.0  0.9580020  0.9533005
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 5e-07 and C = 2.

plot(fit.svm)

#############################
#6. Final Model Predictions #
#############################

#6.1 We will use the model with sigma = 5e-07 and C = 2 for predicting test data
final_model<- predict(fit.svm, test)
#6.2 Evaluation of the model
confusionMatrix(final_model,test$Digit)

# Output:-
#Overall Statistics of the model
#Accuracy : 0.9652     
#95% CI : (0.9614, 0.9687)
#No Information Rate : 0.1135          
#P-Value [Acc > NIR] : < 2.2e-16     


#Hence this RBF model can predict handwritten digits with 96.52% accuracy,which is the highest accuracy so far.
# This model will find various use cases like read invoices/bills, scan parking tickets etc.







