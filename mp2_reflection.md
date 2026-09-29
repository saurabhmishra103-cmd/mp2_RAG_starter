# MP2 Reflection

## What worked

The end-to-end RAG pipeline worked successfully from corpus ingestion through retrieval and answer generation. I used paragraph-based chunking with a target chunk size of 500 characters and reserved a section value with each chunk. This helped the retriever maintain story and section context.

The validation results were particularly encouraging for source retrieval. All 5 questions cited the expected source: 2/2 predefined questions and 3/3 learner questions passed the source-match check.

The final pipeline successfully loaded 5 stories, created 69 chunks, generated embeddings using text-embedding-3-small, stored them in Qdrant, retrieved relevant chunks, and generated answers using gpt-4o-mini with source citations.

## What didn't work

The main issue I encountered was that retrieval quality was not always enough to provide all the information needed to answer a question completely.

The validation output also showed that the expected-fact matches were lower than the source-match results. The source-match was 5/5, while the reported fact matches varied across the five questions. This showed that retrieving the correct document does not necessarily mean that the retrieved chunks contain every fact required to answer a question.

## What I'd change

With another five hours, I would experiment with improving retrieval.

I would test different values of k, improve chunk overlap so that information near chunk boundaries is preserved, and compare the results with a hybrid retrieval or reranking approach. I would also investigate whether the section-based chunking could be improved for questions that require information from multiple parts of a story.

The goal would be to improve not only source retrieval but also the amount of supporting evidence available to the LLM for each answer.

## One surprise

One thing that surprised was how well dense retrieval was able to distinguish between the five different Sherlock Holmes stories. Even though the corpus contains similar narrative language and characters, all five validation questions retrieved and cited the expected story.

At the same time, the project showed that successful RAG is more than finding the correct document. The quality, size, and context of the retrieved chunks have a direct impact on the quality and completeness of the final answer.