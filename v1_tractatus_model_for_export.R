# load libraries
library(tm)          # text mining
library(topicmodels) # topic modeling
library(tidytext)    # text manipulation
library(tibble)      # data frames
library(tidyr)       # data tidying
library(dplyr)       # data manipulation
library(igraph)      # network analysis
library(tidygraph)   # tidy graph manipulation
library(ggraph)      # graph visualization
library(tidyverse)   # tidyverse tools
library(visNetwork)  # interactive network visualization

#### step 1: load and preprocess the text ####

# load the .md file
tractatus <- read_file("Tractatus Logico-Philosophicus.md")

# split the file into paragraphs
documents <- unlist(strsplit(tractatus, "\n\n"))  # split by double newlines
documents <- documents[documents != ""]           # remove empty lines

# convert to a data frame
tractatus_dat <- data.frame(doc_id = 1:length(documents), text = documents, stringsAsFactors = FALSE)

# inspect the data
head(tractatus_dat)

#### step 2: preprocess the text ####

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

#### step 3: create a document-term matrix (dtm) ####

# create a document-term matrix (dtm)
dtm <- DocumentTermMatrix(tractatus_corpus)

# remove sparse terms
dtm <- removeSparseTerms(dtm, sparse = 0.90)  # keep more terms for richer insights

#### step 4: train an lda model ####

# train an lda model
lda_model <- LDA(dtm, k = 5, control = list(seed = 123))  # k = number of topics

#### step 5: extract term-topic distributions ####

# extract term-topic distributions
term_topic <- tidy(lda_model, matrix = "beta")

# filter top terms for each topic (keep top 10 terms per topic)
top_terms <- term_topic %>%
  group_by(topic) %>%
  top_n(10, beta) %>%  # keep top 10 terms per topic
  ungroup()

#### step 6: create a term-term similarity matrix ####

# reshape to wide format (terms x topics) and set terms as row names
term_matrix <- top_terms %>%
  spread(topic, beta, fill = 0) %>%  # reshape to wide format
  column_to_rownames("term") %>%     # set terms as row names
  as.matrix()

# compute cosine similarity between terms
cosine_sim <- function(x) {
  x <- x / sqrt(rowSums(x^2))  # normalize rows
  return(x %*% t(x))           # compute dot product
}

term_sim <- cosine_sim(term_matrix)

# set a moderate similarity threshold (e.g., 0.3)
similarity_threshold <- 0.3
term_sim[term_sim < similarity_threshold] <- 0  # filter out low similarities

#### step 7: create and simplify the term-term network ####

# create a term-term network
term_graph <- graph_from_adjacency_matrix(term_sim, mode = "undirected", weighted = TRUE)

# simplify the graph
term_graph <- simplify(term_graph)

# remove weak edges (e.g., weight < 0.2)
term_graph <- delete_edges(term_graph, E(term_graph)[weight < 0.2])

# remove nodes with low degree (e.g., degree < 3)
term_graph <- delete_vertices(term_graph, V(term_graph)[degree(term_graph) < 3])

#### step 8: add topic information to nodes ####

# add topic information to nodes
top_terms_list <- top_terms %>%
  group_by(term) %>%
  summarize(topics = paste(topic, collapse = ", "))  # list topics for each term

# add topic information to the graph
V(term_graph)$topics <- top_terms_list$topics[match(V(term_graph)$name, top_terms_list$term)]

#### step 9: visualize the term-term network ####

# visualize the term-term network using a faster layout algorithm (e.g., "kk")
set.seed(123)  # for reproducible layout
ggraph(term_graph, layout = "kk") +  # use kamada-kawai layout for speed
  geom_edge_link(aes(width = weight), color = "gray", alpha = 0.7) +
  geom_node_point(aes(color = topics), size = 5) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3) +
  scale_edge_width(range = c(0.5, 2)) +
  theme_void() +
  labs(title = "semantic relations between terms in topics",
       subtitle = "edges represent cosine similarity between terms",
       color = "topics",
       edge_width = "similarity")

#### step 10: interactive term network ####

# convert igraph object to visnetwork format
vis_data <- toVisNetworkData(term_graph)

# extract nodes and edges
nodes <- vis_data$nodes
edges <- vis_data$edges

# customize nodes
nodes$title <- nodes$topics  # show topics on hover
nodes$label <- nodes$name    # show term names as labels
nodes$color <- "lightblue"   # set node color
nodes$size <- 10             # set node size

# customize edges
edges$width <- edges$weight * 1   # scale edge width by weight
edges$color <- "gray"            # set edge color
edges$smooth <- TRUE             # smooth edges for better visualization

# create the interactive network graph
graph <- visNetwork(nodes, edges) %>%
  visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%
  visPhysics(
    enabled = TRUE,  # keep physics enabled
    solver = "forceAtlas2Based",  # use forceAtlas2Based solver
    forceAtlas2Based = list(
      gravitationalConstant = -50,  # adjust repulsion between nodes
      centralGravity = 0.01,        # pull nodes toward the center
      springLength = 100,           # adjust the length of edges
      springConstant = 0.08,        # adjust the stiffness of edges
      damping = 0.4,                # control the speed of the simulation
      avoidOverlap = 1              # prevent nodes from overlapping
    ),
    stabilization = list(
      enabled = TRUE,               # enable stabilization
      iterations = 1000,            # number of iterations to stabilize
      updateInterval = 10           # update interval during stabilization
    )
  ) %>%
  visInteraction(
    dragNodes = TRUE,  # allow dragging nodes
    dragView = TRUE,   # allow dragging the entire graph
    zoomView = TRUE    # allow zooming
  ) %>%
  visLayout(randomSeed = 123) %>%  # set seed for reproducible layout
  # add a title using HTML and CSS
  htmlwidgets::prependContent(
    htmltools::tags$div(
      style = "font-family: Arial; font-size: 20px; text-align: center; margin-bottom: 20px;",
      "Tractatus Term Network"
    )
  ) %>%
  # disable physics after stabilization to prevent rotation
  visEvents(
    stabilizationIterationsDone = "function() {
      this.setPhysicsEnabled(false);  // disable physics after stabilization
    }"
  )

# export the graph as an HTML file
saveWidget(graph, file = "Tractatus_Term_Network.html", selfcontained = TRUE)