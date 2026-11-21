# What is retrieval-augmented generation?

Retrieval-augmented generation (RAG) is a pattern for answering questions with a language model while grounding the answer in documents you provide.

A typical RAG pipeline has four steps. First, you split source documents into chunks. Second, you index those chunks so they can be searched, often with embeddings and a vector database. Third, at question time, you retrieve the chunks most relevant to the question. Fourth, you send the question plus the retrieved text to a language model and ask it to answer only from that evidence.

RAG is useful when the facts live in your files, your wiki, or your product manuals, and are not reliably stored in the model's training data. If the problem is mainly style or format rather than missing knowledge, fine-tuning or structured prompting may be a better fit than RAG.

A careful RAG system cites its sources, says "I don't know" when retrieval is weak, and is tested with a fixed set of questions whose expected answers you already know.
