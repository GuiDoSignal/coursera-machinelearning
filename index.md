
---
title: "Human Activity Recognition Predictor"
author: "Guilherme Junqueira"
date: "September 17, 2016"
output: html_document
---

# Summary

The aim of this project is to build a model that try to classify the exercises 
performed by six young adults in five different ways: exactly according to the 
specification (Class A), throwing the elbows to the front (Class B), 
lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway 
(Class D) and throwing the hips to the front (Class E).

To achieve this goal, we trained a random forest model and evaluated it through 
a 5-fold repeated cross validation with 3 runs.

We show how different values of randow predictors influence the and achieved 
the accuracy of 99.41% with in the training data.

# Pre processing



Firstly, we get the files online. During the initial exploration of the data, it 
was noted that some columns have the string '\#DIV/0!' which probably indicates 
a division by zero error. This leads to some numeric columns being mistakenly 
treated as character columns, so we remove this string and the double quotes 
from the file.


```r
file_train <- "./train_file.csv"
file_test  <- "./test_file.csv"

# Only download it if we don't have it.
if( !file.exists(file_train) ){
    # gets the filse online
    u1 <- url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
    u2 <- url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
    f1 <- readLines(u1)
    f2 <- readLines(u2)
    close(u1)
    close(u2)
    
    # removes double quotes and the string #DIV/0!
    c1 <- gsub( "(\")|(#DIV/0!)", "", f1)
    c2 <- gsub( "(\")|(#DIV/0!)", "", f2)

    # save it to disk
    writeLines(c1, file_train)
    writeLines(c2, file_test)
}

# Load to data frame
data_train <- read.csv(file_train, na.strings = c("NA") )
data_test  <- read.csv(file_test, na.strings = c("NA") )

# Remove files' names
rm(file_test, file_train)

dim(data_train)
```

```
## [1] 19622   160
```

As we can see above, there are 19622 rows and 160 columns in the training data 
set. Some of the rows may not contribute to the model either because they are 
not related to the sensors used or because they have a high proportion (>90%) 
of missing values.

Those columns are removed in the section below and the datasets are reduced to 
53 columns with better quality data. 


```r
# Removing variables unrelated to the sensors
data_train <- subset(
    data_train,
    select = -c(X, new_window, num_window,
                raw_timestamp_part_1, raw_timestamp_part_2,
                user_name, cvtd_timestamp)
)
data_test <- subset(
    data_test,
    select = -c(X, new_window, num_window,
                raw_timestamp_part_1, raw_timestamp_part_2,
                user_name, cvtd_timestamp)
)

# Removing columns with more than 90% missing values
mostly_data <- apply(!is.na(data_train), 2, sum) > (0.90 * nrow(data_train))

data_train <- data_train[, mostly_data]
data_test  <- data_test[, mostly_data]

# Removing temporary objects
rm(mostly_data)

dim(data_train)
```

```
## [1] 19622    53
```

# Model selection

Now that we have better data, we will train and evaluate our model. In the same 
fashion of the [original work][originalwork], we chose a random forest predictor 
because of the inherent noise present in the data and the likely feature 
selection provided by the algorithm.


```r
# For reproducibility reasons
set.seed(2016)
k_fold       <- 5
num_repeats  <- 3
seeds_length <- k_fold * num_repeats

seeds_list   <- vector(mode = "list", length = seeds_length + 1)
for(i in 1:seeds_length){
    seeds_list[[i]] <- sample.int(1000, num_repeats)
}
seeds_list[[seeds_length + 1]] <- sample.int(1000, 1)

# Tuning training options
train_ctrl <- trainControl(
    method  = "repeatedcv",
    repeats = num_repeats, 
    number  = k_fold,
    seeds   = seeds_list
)

# Train model
set.seed(2017)
model_rf <- train(
    classe ~ .,
    data_train,
    method = "parRF",
    trControl = train_ctrl
)
```

```
## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info =
## trainInfo, : There were missing values in resampled performance measures.
```

As you can see on the code above, we did a 5-fold repeated cross validation 
with 3 runs to evaluate our model. The chart below shows that we achieved 
99.41% of accuracy with 27 random predictors select at each split (mtry = 27).

![plot of chunk acc1](figure/acc1-1.png)

```
## 
## Call:
##  randomForest(x = "x", y = "y", ntree = 125, mtry = 27) 
##                Type of random forest: classification
##                      Number of trees: 250
## No. of variables tried at each split: 27
```

It is important to note that this accuracy is optmistic, so we expected to 
achieve lower values when applying this model to the test data.  

The chart below shows the confusion matrix obtained from the training data as a 
heat map.

![plot of chunk conf-mat1](figure/conf-mat1-1.png)

[originalwork]: http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf "Qualitative Activity Recognition of Weight Lifting Exercises"
