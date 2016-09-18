# Human Activity Recognition Predictor

Guilherme Junqueira  
September 18, 2016  

---

## Summary

The aim of this project is to build a model that try to classify the exercises 
performed by six young adults in five different ways: exactly according to the 
specification (Class A), throwing the elbows to the front (Class B), 
lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway 
(Class D) and throwing the hips to the front (Class E).

To achieve this goal, we trained a random forest model with 10 different values 
for the number of randomly selected predictors (mtry) and evaluated them through 
a 5-fold repeated cross validation with 3 runs.

Finally, we showed how these different values influence the accuracy and the 
confusion matrix obtained in the final model, whose accuracy was higher than 
99.41% with in the training data.

## Pre processing



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
data_train  <- data_train[, mostly_data]
data_test   <- data_test[, mostly_data]

dim(data_train)
```

```
## [1] 19622    53
```

## Model selection

Now that we have better data, we will train and evaluate our model. In the same 
fashion of the [original work][originalwork], we chose a random forest predictor 
because of the inherent noise present in the data and the likely feature 
selection provided by the algorithm.


```r
# For reproducibility reasons
set.seed(2016)
k_fold       <- 5
num_repeats  <- 3
tune_length  <- 10
seeds_length <- k_fold * num_repeats

seeds_list   <- vector(mode = "list", length = seeds_length + 1)
for(i in 1:seeds_length){
    seeds_list[[i]] <- sample.int(1000, tune_length)
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
    tuneLength = tune_length,
    trControl = train_ctrl
)
```

```
## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info =
## trainInfo, : There were missing values in resampled performance measures.
```

As you can see on the code above, we tried 10 different values for the mtry
parameter (number of random selected predictors at each split) and evaluated 
each of them with a 5-fold repeated cross validation with 3 runs. 

The chart below shows that the accuracy of 99.4% was achieved when mtry = 27.


```
## Parallel Random Forest 
## 
## 19622 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 3 times) 
## Summary of sample sizes: 15697, 15699, 15696, 15697, 15699, 15698, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9935344  0.9918211
##    7    0.9956863  0.9945435
##   13    0.9956399  0.9944848
##   18    0.9951160  0.9938221
##   24    0.9946488  0.9932309
##   29    0.9935616  0.9918555
##   35    0.9931567  0.9913432
##   40    0.9921008  0.9900076
##   46    0.9913646  0.9890761
##   52    0.9891705  0.9863003
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 7.
```

![plot of chunk acc1](figure/acc1-1.png)

It is important to note that this accuracy is optmistic, so we expected to 
achieve lower values when applying this model to the test data.  

The chart below shows the confusion matrix obtained from the training data as a 
heat map.

![plot of chunk conf-mat1](figure/conf-mat1-1.png)

[originalwork]: http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf "Qualitative Activity Recognition of Weight Lifting Exercises"


