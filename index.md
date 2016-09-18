
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

We estimate tour model 


```r
# First load needed libraries
require(caret)
require(ggplot2)
require(doParallel)
registerDoParallel()

# https://bigcomputing.blogspot.com.br/2014/10/an-example-of-using-random-forest-in.html
```

# Model selection

# Cross validation

# Prediction

Firstly, we get the file online. During the initial exploration of the data, it 
was noted that some columns have the string '\#DIV/0!' which probably a division 
by zero error. This leads to some numeric columns being mistakenly treated as 
character columns, so we remove this string and the double quotes from the file.


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

Now that we have better data, we will train and evaluate our model. In the same 
fashion of the [original work][originalwork], we chose a random forest predictor 
because of the inherent noise present in the data and the likely feature 
selection provided by it. 


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

print(model_rf)
```

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
##    2    0.9937995  0.9921565
##   27    0.9940882  0.9925218
##   52    0.9886690  0.9856660
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

To evaluate our model, we used a 


```r
conf_matrix     <- confusionMatrix(model_rf, norm="overall")
data_confMatrix <- data.frame(conf_matrix$table) 
p               <- ggplot(data_confMatrix)
p               <- p + geom_tile(aes(x=Reference, y=Prediction, fill=Freq))
p               <- p + scale_x_discrete(name = "Actual Class")
p               <- p + scale_y_discrete(name = "Predicted Class")
p               <- p + scale_fill_gradient(breaks = seq(from=0, to=100, by=10))
p               <- p + labs(fill="Normalized\nFrequency")
p
```

![plot of chunk unnamed-chunk-1](figure/unnamed-chunk-1-1.png)

[originalwork]: http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf "Qualitative Activity Recognition of Weight Lifting Exercises"
