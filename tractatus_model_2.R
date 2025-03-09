# load necessary libraries
library(tm)          # text mining
library(topicmodels) # lda topic modeling
library(tidytext)    # tidy text analysis
library(igraph)      # network visualization
library(tidyr)       # data reshaping
library(ggplot2)     # visualization
library(igraph)      # network visualization
library(tidygraph)   # tidy graph manipulation
library(ggraph)      # ggplot2-like graph visualization
library(dplyr)       # data manipulation
library(tibble)      # for matrix gen  

# load the .md file
tractatus <- read_file("Tractatus Logico-Philosophicus.md")

# split the file into paragraphs
documents <- unlist(strsplit(tractatus, "\n\n"))  # split by double newlines
documents <- documents[documents != ""]           # remove empty lines

# convert to a data frame
tractatus_dat <- data.frame(doc_id = 1:length(documents), text = documents, stringsAsFactors = FALSE)

# inspect the data
head(tractatus_dat)

# convert to a corpus
tractatus_corpus <- Corpus(VectorSource(tractatus_dat$text))

# preprocess the text
tractatus_corpus <- tm_map(tractatus_corpus, content_transformer(tolower))  # convert to lowercase
tractatus_corpus <- tm_map(tractatus_corpus, removePunctuation)             # remove punctuation
tractatus_corpus <- tm_map(tractatus_corpus, removeWords, stopwords("german")) # remove german stopwords
tractatus_corpus <- tm_map(tractatus_corpus, stripWhitespace)               # remove extra whitespace
tractatus_corpus <- tm_map(tractatus_corpus, removeNumbers)                 # remove numbers

# handle umlauts
tractatus_corpus <- tm_map(tractatus_corpus, content_transformer(function(x) {
  x <- gsub("ä", "ae", x)
  x <- gsub("ö", "oe", x)
  x <- gsub("ü", "ue", x)
  x <- gsub("ß", "ss", x)
  return(x)
}))

# create a document-term matrix (dtm)
dtm <- DocumentTermMatrix(tractatus_corpus)

# remove sparse terms
dtm <- removeSparseTerms(dtm, sparse = 0.95)

# train an lda model
lda_model <- LDA(dtm, k = 5, control = list(seed = 123))  # k = number of topics

# extract topic-term distributions
topic_term <- tidy(lda_model, matrix = "beta")

# reshape the data into a wide format (topics x terms)
topic_term_wide <- topic_term %>%
  spread(term, beta, fill = 0)

# convert to a matrix
topic_matrix <- as.matrix(topic_term_wide[, -1])  # remove the topic column
rownames(topic_matrix) <- topic_term_wide$topic

# compute cosine similarity
cosine_sim <- function(x) {
  x <- x / sqrt(rowSums(x^2))  # normalize rows
  return(x %*% t(x))           # compute dot product
}

topic_sim <- cosine_sim(topic_matrix)

# create a topic similarity network
topic_graph <- graph_from_adjacency_matrix(topic_sim, mode = "undirected", weighted = TRUE)

# simplify the graph
topic_graph <- simplify(topic_graph, remove.multiple = TRUE, remove.loops = TRUE)

# visualize the network
plot(topic_graph,
     layout = layout_with_fr(topic_graph),
     vertex.size = degree(topic_graph) * 10,
     vertex.color = "lightblue",
     vertex.label.cex = 0.8,
     vertex.label.color = "black",
     edge.width = E(topic_graph)$weight * 3,
     edge.color = "gray",
     main = "topic similarity network")

# interpretation: topics are well-connected and semantically similar 

# compute topic-term distribution network 

#  extract term-topic distributions
term_topic <- tidy(lda_model, matrix = "beta")

# filter top terms for each topic
top_terms <- term_topic %>%
  group_by(topic) %>%
  top_n(10, beta) %>%  # select top 10 terms per topic
  ungroup()

# create a term-term similarity matrix
term_matrix <- term_topic %>%
  spread(topic, beta, fill = 0) %>%  # reshape to wide format (terms x topics)
  column_to_rownames("term") %>%     # set terms as row names
  as.matrix()

# compute cosine similarity between terms
cosine_sim <- function(x) {
  x <- x / sqrt(rowSums(x^2))  # normalize rows
  return(x %*% t(x))           # compute dot product
}

term_sim <- cosine_sim(term_matrix)

# set a similarity threshold (e.g., 0.2)
similarity_threshold <- 0.2
term_sim[term_sim < similarity_threshold] <- 0  # filter out low similarities

# create a term-term network
term_graph <- graph_from_adjacency_matrix(term_sim, mode = "undirected", weighted = TRUE)

# simplify the graph
term_graph <- simplify(term_graph, remove.multiple = TRUE, remove.loops = TRUE)

# add topic information to nodes
top_terms_list <- top_terms %>%
  group_by(term) %>%
  summarize(topics = paste(topic, collapse = ", "))  # list topics for each term

# add topic information to the graph
V(term_graph)$topics <- top_terms_list$topics[match(V(term_graph)$name, top_terms_list$term)]

# visualize the term-term network
set.seed(123)  # for reproducible layout
ggraph(term_graph, layout = "fr") +
  geom_edge_link(aes(width = weight), color = "gray", alpha = 0.7) +
  geom_node_point(aes(color = topics), size = 5) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3) +
  scale_edge_width(range = c(0.5, 2)) +
  theme_void() +
  labs(title = "semantic relations between terms in topics",
       subtitle = "edges represent cosine similarity between terms",
       color = "topics",
       edge_width = "similarity")





