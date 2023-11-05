#load libraries
library(caret)
library(quanteda)
library(rpart)
library(e1071)

#proportion to sample
sample_proportion <- 0.05

#load the original dataset
imdb_data <- read.csv("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/imdb.csv", header = TRUE)

#sample a subset of the original dataset
num_samples <- round(nrow(imdb_data) * sample_proportion)
sample_indices <- sample(1:nrow(imdb_data), size = num_samples)
sample_data <- imdb_data[sample_indices, ]

#split data into training and testing subsets
set.seed(123)
indexes <- createDataPartition(sample_data$sentiment, p = 0.7, list = FALSE)
training_data <- sample_data[indexes, ]
testing_data <- sample_data[-indexes, ]

#save training and testing data
write.csv(training_data, file = paste("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/training_data.csv", sep = ""), row.names = FALSE)
write.csv(testing_data, file = paste("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/testing_data.csv", sep = ""), row.names = FALSE)

#word2vec big trainign set (everything except testing_data)
imdb_data_without_testing <- imdb_data[!imdb_data$review   %in% testing_data$review, ]
write.csv(imdb_data_without_testing, file = paste("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/imdb_data_without_testing.csv", sep = ""), row.names = FALSE)
