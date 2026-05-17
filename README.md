# Part 3: NLP and Sequence Modeling Mini Project

This milestone documents the implementation of a natural language processing architecture designed to categorize textual customer support communications by underlying sentiment states.

## Dataset Profile & Structural Explorations
* **Total Clean Text Ingest Rows:** 1,500 rows.
* **Feature Dimension Vectors:** 6 base tracking dimensions (`ticket_id`, `channel`, `customer_message`, `sentiment_label`, `word_count`, `urgent_flag`).
* **Target Classification Classes:** Highly balanced categorical layout:
  * `neutral`: 524 rows (34.93%)
  * `negative`: 497 rows (33.13%)
  * `positive`: 479 rows (31.93%)
* **Data Integrity Checks:** Confirmed 0 null fields across the entire text tracking matrix.

---

## Text Preprocessing & Cleaning Pipelines
To eliminate syntax variance and vocabulary inflation, text blobs are routed through a destructive cleaning engine:
1. **Case Standardization:** Forced all character structures to lowercase arrays to prevent the vectorizer from treating words like "Refund" and "refund" as distinct tokens.
2. **Entity Token Stripping:** Used regular expressions (`r'ticket number is \d+'`) to drop specific alphanumeric noise metrics that have zero correlation with semantic intent.
3. **Punctuation Stripping:** Cleared non-alphanumeric noise barriers to prevent punctuation strings from clustering onto word stems.
4. **Token Isolation & Trailing Compression:** Stripped edge spaces and compressed loose inner whitespace sequences into a single space marker.
5. **Stratified Matrix Splitting:** Partitioned dataset matrices using an 80/20 stratified split configuration based on true labels to lock in exact distribution balance across both training and test boundaries.

---

## Vectorization Strategy & Document Properties
Raw text strings were transformed using a Term Frequency-Inverse Document Frequency (`TF-IDF`) vectorization matrix strategy:
* **Feature Scope Boundary:** Restricted arrays to the top 1,000 foundational feature terms to manage the curse of dimensionality and prevent sparse matrix bloat.
* **Gram Expansion Split:** Configured an `ngram_range=(1, 2)` bounds definition to capture both individual vocabulary words (unigrams) and essential consecutive phrase markers like "not working" or "fast refund" (bigrams).
* **Lexical Stop-Word Exclusion:** Scrubbed generic English functional pronouns and articles using a baseline stop-words filter to keep model focus strictly on semantic descriptors.

---

## Technical Deep-Dive: Evolution of Sequence Modeling Architectures

To build high-performance language systems, it is vital to trace how sequential architectures evolve past the core limitations of early networks:

### 1. Recurrent Neural Networks (RNNs)
* **Structural Architecture:** RNNs process language linearly by iterating through word vectors one step at a time. They pass a hidden state vector forward through a feedback chain loop to serve as an internal memory footprint of past tokens.
* **Core Computational Failures:** RNNs suffer heavily from the **Vanishing Gradient Problem**. During backpropagation across long text strings, gradients are repeatedly multiplied by weight matrices. If those values are fractional, the gradient shrinks exponentially toward zero, preventing the network from updating weights for early tokens. This means RNNs fail completely at capturing long-term contextual dependencies in multi-sentence messages.

### 2. Long Short-Term Memory Networks (LSTMs)
* **Structural Architecture:** LSTMs overcome short-term retention barriers by replacing the simple hidden state layer with a specialized **Cell State** memory track. This lane is regulated by three distinct mathematical gates:
  * **Forget Gate:** Decides what information from previous steps to drop using a sigmoid layer.
  * **Input Gate:** Selects which new feature elements from the current token modify the memory cell.
  * **Output Gate:** Determines the next hidden state based on the newly updated cell state data.
* **Core Computational Failures:** While gates protect the error gradient across longer sentences, LSTMs remain **fundamentally linear**. Because step $t$ cannot run until step $t-1$ finishes, training cannot be parallelized across GPU hardware, creating massive scalability bottlenecks on large corporate text volumes.

### 3. Attention Mechanisms
* **Structural Architecture:** Instead of forcing a network to compress an entire sentence into a single, fixed-size hidden vector, Attention allows the model to look back at all historical input positions simultaneously. 
* **Core Computational Failures:** It assigns dynamic relevance scalars (attention weights) to different tokens based on current evaluation points. This lets the network focus on specific word groups (such as linking a pronoun directly to a product mentioned three lines prior) regardless of their distance from one another. However, when paired with an LSTM backbone, it still faces sequential training slowdowns.

### 4. Transformers
* **Structural Architecture:** Transformers abandon recurrent loops entirely, utilizing a mechanism known as **Self-Attention** alongside **Positional Encodings** to process all tokens in a text block completely in parallel.
* **Core Advancements & Advantages:** By processing entire paragraphs simultaneously, Transformers maximize GPU cluster utilization, cutting training times down drastically. Every word calculates its relationship to every other word across the text block at the same time. This structure easily models long-range contextual relationships across documents, which explains why it serves as the foundational backbone for all modern LLMs and advanced enterprise text solutions.