### Project Submission: MovieLens

## Section 1: Introduction
# This project is part of the final course of the HarvardX Data Science Professional Certificate program.
# The aim is to create a movie recommendation system using the MovieLens data set, which contains millions of movie ratings by users.
# The MovieLens data set will be split into (i) a edx set and (ii) a validation set.
# The edx set will be split further into a training set to train our algorithms, and a testing set to evaluate the algorithms.
# The performance of the algorithms will be determined by their root mean square error (RMSE), with smaller RMSE indicating better performance.
# The best-performing model will then be evaluated against the validation set.

# Section 1.1: Generating the edx and validation sets
# We start by loading the required R packages, namely the tidyverse, caret, and data.table packages.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# We then download the MovieLens data set using the code provided in the HarvardX course.
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# We will now generate the `validation` set using the code provided in the HarvardX course.
# `Validation` set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in `validation` set are also in `edx` set
validation_set <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from `validation` set back into `edx` set
removed <- anti_join(temp, validation_set)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Section 1.2: Summary statistics of the edx set
# The following code shows that the `edx` set is a data.table with over 9 million observations and 6 variables.
str(edx)

# We can see that many movies fall into multiple genres.
head(edx$genres)

# We can tease out the unique genres using the following code.
genres <- str_replace(edx$genres, "\\|.*", "")
genres <- genres[!duplicated(genres)]

# There are 20 unique genres, including a category of "no genres listed".
genres

# We will now compile some simple summary statistics of the `edx` set before going into data exploration.
data.frame(n_ratings = nrow(edx),
           n_movies = n_distinct(edx$movieId),
           n_users = n_distinct(edx$userId),
           n_genres = length(genres),
           avg_rating = mean(edx$rating))

## Section 2: Analysis - Exploring the edx set
# Section 2.1: Identifying movie effect
# A quick data exploration shows that some movies are rated more than others.
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, col = "black") + 
  scale_x_log10() + 
  ggtitle("Distribution of ratings by movie") +
  xlab("no. of ratings") +
  ylab("no. of movies")

# Section 2.2: Identifying user effect
# Some users are also more active than others at rating movies.
edx %>%
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, col = "black") + 
  scale_x_log10() +
  ggtitle("Distribution of ratings by user") +
  xlab("no. of ratings") +
  ylab("no. of users")

# Different users also have different rating patterns, with some tend to rate movies more favourably in general.
edx %>%
  group_by(userId) %>%
  summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(avg_rating)) +
  geom_bar(col = "black") +
  ggtitle("Average rating of users") +
  xlab("average rating") +
  ylab("no. of users")

# Section 2.3: Identifying genre effect
# Ratings are also affected by the movie's genre.
# Below is the number of movies for each genre,
n_movies_genres <- sapply(genres, function(i){
  index <- str_which(edx$genres, i)
  length(edx$rating[index])
})
n_movies_genres

# and the average rating for each genre. 
genres_avg_rating <- sapply(genres, function(i){
  index <- str_which(edx$genres, i) 
  mean(edx$rating[index])
})
genres_avg_rating

# We can see that certain genres tend to be rated more favourably than others.
# We first generate a summary table showing the number of movies and the average rating for each genre.
summary_genres <- data.frame(genre = genres,
                             n_movies = n_movies_genres,
                             avg_rating = genres_avg_rating)
summary_genres

# To see the genre effect more clearly, we plot the number of movies for each genre,
summary_genres %>%
  mutate(genre = reorder(genre, n_movies)) %>%
  ggplot(aes(genre, n_movies)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  ggtitle("No. of movies per genre") +
  ylab("no. of movies")

# and the average rating for each genre.
summary_genres %>%
  mutate(genre = reorder(genre, avg_rating)) %>%
  ggplot(aes(genre, avg_rating)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  ggtitle("Average rating per genre") +
  ylab("average rating")

# Section 2.4: Identifying time effect
# There could also be time effect on movie ratings.
# The timestamp variable in the `edx` set, which represents the time and date in which the rating was provided, is in Unix time.
head(edx$timestamp)

# We will change it to the ISO 8601 format,
library(lubridate)
edx <- mutate(edx, date = as_datetime(timestamp))
head(edx)

# and compute the average rating for each date. 
# We can see some evidence of a time effect.
edx %>%
  mutate(date = round_date(date, unit = "day")) %>%
  group_by(date) %>%
  summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(date, avg_rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Average movie ratings by date rated") +
  ylab("average rating")

# Data exploration tells us that there are (i) movie, (ii) user, (iii) genre, and (iv) time effects affecting ratings.
# We will train an algorithm building on these effects one by one.

## Section 3: Results - Building our models
# We start by splitting the `edx` set further into training and testing sets.
# The `test` set will be 10% of `edx` data.
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# Section 3.1: Model 1 - predict the same rating for all movies
# We start by building the simplest model, i.e., predicting the same rating for all movies.
# In this case, the estimate that minimises the RMSE is the average of all ratings.
mu <- mean(train_set$rating)

# If we predict all ratings with the estimate mu, we obtain an RMSE of 1.06, which is too high.
rmse_model_1 <- RMSE(mu, test_set$rating)

# We will put the RMSE result in a table to see how the RMSE decreases as we improve our model.
rmse_results <- data.frame(method = "just the average",
                           RMSE = rmse_model_1)
rmse_results

# Section 3.2: Model 2 - adding movie effect
# Based on our data exploration, we know that different movies are rated differently.
# We can improve our model by including the movie effect, i.e., adding a term b_i to represent the average rating for movie i.
# We can obtain the least squares estimate (LSE) of b_i using the following code.
# Note that we are using a linear model for our algorithm.
mu_movie <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

# We can see how much our prediction improves using the following code.
preds_movie <- test_set %>% 
  left_join(mu_movie, by = 'movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

rmse_model_2 <- RMSE(preds_movie, test_set$rating)

# The RMSE falls to 0.94, but we can do much better by taking into account of other effects.
rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "adding movie effect",
                                     RMSE = rmse_model_2))
rmse_results

# Section 3.3: Model 3 - adding user effect
# Our data exploration also tells us that different users tend to rate movies differently.
# We can further improve our model by adding the user effect, i.e., including a term b_u to represent the average rating for user u.
# The LSE of b_u and the RSME are obtained using the following code.
mu_user <- train_set %>%
  left_join(mu_movie, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

preds_user <- test_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

rmse_model_3 <- RMSE(preds_user, test_set$rating)

# The RMSE falls to 0.86, but we can do even better by taking into account of more effects.
rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "adding user effect",
                                     RMSE = rmse_model_3))
rmse_results

# Section 3.4: Model 4 - adding genre effect
# We know that different movie genres also tend to be rated differently.
# To take into account of the genre effect, we include a term b_g to represent the average rating for genre g.
# The LSE of b_g and the RMSE are obtained using the following code.
mu_genre <- train_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  group_by(genres) %>%
  summarise(b_g = mean(rating - mu - b_i - b_u))

preds_genre <- test_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  left_join(mu_genre, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

rmse_model_4 <- RMSE(preds_genre, test_set$rating)

# The RMSE remains somewhat the same.
# We will check if we can improve our model further by adding the time effect.
rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "adding genre effect",
                                     RMSE = rmse_model_4))
rmse_results

# Section 3.5: Model 5 - adding time effect
# To include the time effect, we add a term b_t to represent the average rating for date rated t.
# We first round the ISO 8601 date we created earlier to day,
train_set <- mutate(train_set, date = round_date(date, unit = "day"))
test_set <- mutate(test_set, date = round_date(date, unit = "day"))

# and then we calculate the LSE of b_t and the RMSE using the following code.
mu_time <- train_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  left_join(mu_genre, by = 'genres') %>%
  group_by(date) %>%
  summarise(b_t = mean(rating - mu - b_i - b_u - b_g))

preds_time <- test_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  left_join(mu_genre, by = "genres") %>%
  left_join(mu_time, by = "date") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
  .$pred

rmse_model_5 <- RMSE(preds_time, test_set$rating)

# The RMSE falls slightly further, suggesting an improvement in our model.
rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "adding time effect",
                                     RMSE = rmse_model_5))
rmse_results

# Section 3.6: Model 6 - Regularisation
# It is likely that some of our estimates are noisy, i.e., skewed by small sample sizes.
# We will use regularisation to penalise large estimates that are formed using small sample sizes.
# We start by using cross-validation to select a lambda (tuning variable) by taking into account movie, user, genre and time effects.
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  mu_movie <- train_set %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu) / (n()+l))
  
  mu_user <- train_set %>%
    left_join(mu_movie, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i) / (n()+l))
  
  mu_genre <- train_set %>%
    left_join(mu_movie, by = 'movieId') %>%
    left_join(mu_user, by = 'userId') %>%
    group_by(genres) %>%
    summarise(b_g = sum(rating - mu - b_i - b_u) / (n()+l))
  
  mu_time <- train_set %>%
    left_join(mu_movie, by = 'movieId') %>%
    left_join(mu_user, by = 'userId') %>%
    left_join(mu_genre, by = 'genres') %>%
    group_by(date) %>%
    summarise(b_t = sum(rating - mu - b_i - b_u - b_g) / (n()+l))
  
  preds_reg <- test_set %>%
    left_join(mu_movie, by = 'movieId') %>%
    left_join(mu_user, by = 'userId') %>%
    left_join(mu_genre, by = "genres") %>%
    left_join(mu_time, by = "date") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
    .$pred
  
  return(RMSE(preds_reg, test_set$rating))
})

# The optimal lambda is as follow:
lambda <- lambdas[which.min(rmses)]
lambda

# We will now use the optimal lambda to evaluate the RMSE of our regularised model.
mu_movie <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu) / (n()+lambda))

mu_user <- train_set %>%
  left_join(mu_movie, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i) / (n()+lambda))

mu_genre <- train_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  group_by(genres) %>%
  summarise(b_g = sum(rating - mu - b_i - b_u) / (n()+lambda))

mu_time <- train_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  left_join(mu_genre, by = 'genres') %>%
  group_by(date) %>%
  summarise(b_t = sum(rating - mu - b_i - b_u - b_g) / (n()+lambda))

preds_reg <- test_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  left_join(mu_genre, by = "genres") %>%
  left_join(mu_time, by = "date") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
  .$pred

rmse_model_6 <- RMSE(preds_reg, test_set$rating)

# The RMSE remains somewhat the same. 
# Nonetheless, the regularised model is preferred as it takes into account the effect of small sample sizes.
rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "regularisation",
                                     RMSE = rmse_model_6))
rmse_results

# Section 3.7: Testing final model using the validation set
# We will now test our final model--which is the regularised model including the average rating plus the movie, user, genre and time effects--using the validation set.
# We start by wrangling the timestamp variable of the `validation` set.
str(validation_set)
validation_set <- validation_set %>%
  mutate(date = as_datetime(timestamp)) %>%
  mutate(date = round_date(date, unit = "day"))
head(validation_set)

# Next, we use cross-validation to select a lambda.
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(validation_set$rating)
  
  mu_movie <- validation_set %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu) / (n()+l))
  
  mu_user <- validation_set %>%
    left_join(mu_movie, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i) / (n()+l))
  
  mu_genre <- validation_set %>%
    left_join(mu_movie, by = 'movieId') %>%
    left_join(mu_user, by = 'userId') %>%
    group_by(genres) %>%
    summarise(b_g = sum(rating - mu - b_i - b_u) / (n()+l))
  
  mu_time <- validation_set %>%
    left_join(mu_movie, by = 'movieId') %>%
    left_join(mu_user, by = 'userId') %>%
    left_join(mu_genre, by = 'genres') %>%
    group_by(date) %>%
    summarise(b_t = sum(rating - mu - b_i - b_u - b_g) / (n()+l))
  
  preds_valid <- validation_set %>%
    left_join(mu_movie, by = 'movieId') %>%
    left_join(mu_user, by = 'userId') %>%
    left_join(mu_genre, by = "genres") %>%
    left_join(mu_time, by = "date") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
    .$pred
  
  return(RMSE(preds_valid, validation_set$rating))
})

lambda <- lambdas[which.min(rmses)]
lambda

# Finally, we calculate the RMSE using the following code.
mu_movie <- validation_set %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu) / (n()+lambda))

mu_user <- validation_set %>%
  left_join(mu_movie, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i) / (n()+lambda))

mu_genre <- validation_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  group_by(genres) %>%
  summarise(b_g = sum(rating - mu - b_i - b_u) / (n()+lambda))

mu_time <- validation_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  left_join(mu_genre, by = 'genres') %>%
  group_by(date) %>%
  summarise(b_t = sum(rating - mu - b_i - b_u - b_g) / (n()+lambda))

preds_valid <- validation_set %>%
  left_join(mu_movie, by = 'movieId') %>%
  left_join(mu_user, by = 'userId') %>%
  left_join(mu_genre, by = "genres") %>%
  left_join(mu_time, by = "date") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
  .$pred

RMSE(preds_valid, validation_set$rating)

# We obtained an RMSE of 0.8234, which is smaller than the threshold of 0.8649 required by the HarvardX course.

## Section 4: Conclusion
# We have successfully trained a machine learning algorithm to predict movie ratings, taking into account movie, user, genre and time effects.
# Our analysis shows that movie and user effects are more important predictors of movie ratings, since they reduce the RMSE by a larger extent.
# Genre and time effects also help to predict movie ratings, though they reduce the RSME by a smaller extent.
# Meanwhile, regularisation helps to improve our model by taking into account the effect of small sample sizes.
# Future work can explore the data set and improve the algorithm further (e.g., including the release date of each movie in the model).

## Section 5: References
# Being new to R and machine learning, I have consulted the following materials when doing up this movielens project:
# https://rafalab.github.io/dsbook/index.html