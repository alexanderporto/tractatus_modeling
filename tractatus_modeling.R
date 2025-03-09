# install packages
# install.packages("tm")          # text mining
# install.packages("tidytext")    # tidy text analysis
# install.packages("topicmodels") # LDA modeling
# install.packages("dplyr")       # data manipulation
# install.packages("ggplot2")     # visualization
# install.packages("readr")       # reading text files
# install.packages("SnowballC")   # stemming for German
# install.packages("widyr")       # for semantic network 

# load libraries
library(tm)
library(tidytext)
library(topicmodels)
library(dplyr)
library(ggplot2)
library(readr)
library(SnowballC)
library(igraph)
library(widyr)


#### load/clean tractatus file ####
tractatus <- read_file("Tractatus Logico-Philosophicus.md")

# split lines 
documents <- unlist(strsplit(tractatus, "n\n")) # split by double new line

documents <- documents[documents != ""] # remove empty lines 

# convert to df 
tractatus_dat <- data.frame(doc_id = 1:length(documents),text=documents, stringsAsFactors = FALSE)

head(tractatus_dat)

#### preprocess ####  

# convert to corpus 
tractatus_corpus <- Corpus(VectorSource(tractatus_dat$text)) # ...$text = text extracted from tracatus, stored as a single df variable 

# preprocess 
tractatus_corpus <- tm_map(tractatus_corpus, content_transformer(tolower)) # converts to lower case 

tractatus_corpus <- tm_map(tractatus_corpus, removePunctuation) # removes punctuation  

tractatus_corpus <- tm_map(tractatus_corpus, removeWords, stopwords("german")) # removes stop-words 

tractatus_corpus <- tm_map(tractatus_corpus, stripWhitespace) # removes extra white space 

tractatus_corpus <- tm_map(tractatus_corpus, removeNumbers) # removes numbers 

# handle umlauts: 
tractatus_corpus <- tm_map(tractatus_corpus, content_transformer(function(x) {
  x <- gsub("ä", "ae", x)
  x <- gsub("ö", "oe", x)
  x <- gsub("ü", "ue", x)
  x <- gsub("ß", "ss", x)
  return(x)
}))


# optional step: stem words to root form: 
# tractatus_corpus <- tm_map(tractatus_corpus, stemDocument, language = "german")

# inspect preprocess result: 
inspect(tractatus_corpus)

#### create a DTM (document-term matrix) #### 
dtm <- DocumentTermMatrix(tractatus_corpus)
inspect(dtm)

#### run LDA model on DTM #### 
set.seed(123)

# LDA with 6 topics. modify k=n to change 
lda_model <- LDA(dtm, k=6, control=list(seed=123))
lda_model

#### explore LDA model #### 
top_terms <- tidy(lda_model, matrix = "beta")

# group by topic: 
top_terms <- top_terms %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# reorder to visualize
top_terms <- top_terms %>%
  mutate(term = reorder(term, beta))

# visualize 
ggplot(top_terms, aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  labs(title = "Top 10 Terms per Topic", x = "Term", y = "Beta")

#### word frequency #### 
word_freq <- colSums(as.matrix(dtm))
word_freq <- sort(word_freq, decreasing = TRUE)

# convert to df 
word_freq_df <- data.frame(word = names(word_freq), freq = word_freq)

# plot top 20 words 
ggplot(head(word_freq_df, 20), aes(x = reorder(word, freq), y = freq)) +
  geom_col(fill = "blue") +
  coord_flip() +
  labs(title = "Top 20 Words by Frequency", x = "Word", y = "Frequency")

#### n-gram analysis #### 
# sequences of words 

# tokenize into bigrams
bigrams <- tractatus_dat %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

# count bigram frequencies 
bigram_counts <- bigrams %>%
  count(bigram, sort = TRUE)

# plot top 10 bigrams 
ggplot(head(bigram_counts, 10), aes(x = reorder(bigram, n), y = n)) +
  geom_col(fill = "blue") +
  coord_flip() +
  labs(title = "Top 10 Bigrams", x = "Bigram", y = "Frequency")

#### network analysis of LDA  #### 

# Convert corpus to a data frame
tractatus_tidy <- data.frame(text = sapply(tractatus_corpus, as.character), stringsAsFactors = FALSE)

# Tokenize words
tractatus_tokens <- tractatus_tidy %>%
  unnest_tokens(word, text)

# Count word frequencies
word_counts <- tractatus_tokens %>%
  count(word, sort = TRUE)

# Filter out words occurring very rarely (e.g., less than 3 times)
common_words <- word_counts %>%
  filter(n > 2)

# Retain only frequent words
tractatus_tokens <- tractatus_tokens %>%
  filter(word %in% common_words$word)

# Define word co-occurrence within a window of 3 words
cooccurrence <- tractatus_tokens %>%
  pairwise_count(word, doc_id, sort = TRUE, upper = FALSE) # from widyr package

# Inspect top co-occurrences
head(cooccurrence)




# 
# 
# # GloVe semantic embedding 
# 
# # download and load embeddings: https://nlp.stanford.edu/projects/glove/ 
# glove_embeddings <- read.table("glove.6B.50d.txt", quote = "", comment.char = "", stringsAsFactors = FALSE)
# 
# 
