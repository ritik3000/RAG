# RAG From the Ground Up

I wanted to actually understand RAG—not just use a framework that abstracts it away.

So I built each layer from scratch on real financial documents (Syrma quarterly filings) 
and compared how retrieval quality changed at each step.

## What I Built & Why

**Naive RAG** — the baseline. Embed chunks, do cosine similarity, pass to LLM.
Simple, but retrieval breaks on keyword-heavy financial queries.

**BM25 + Hybrid Search** — added keyword search alongside semantic search.
Fused both rankings with Reciprocal Rank Fusion (RRF). 
Noticeable improvement on queries like "Q4 FY26 results" where exact terms matter.

**Cross-Encoder Reranking** — semantic search casts a wide net (top 20), 
reranker picks the 3 actually relevant chunks.
Slower, but precision improves meaningfully.

**Full Pipeline** — hybrid retrieval → rerank → generate.
The architecture most production systems actually use.

## What I Observed

- Naive RAG struggles when the query uses different terminology than the document
- RRF fusion consistently outperformed either search method alone
- Reranking had the highest per-query impact but adds latency—worth it for precision-critical use cases

## Stack
OpenAI (embeddings + generation), ChromaDB, BM25Okapi, 
CrossEncoder (ms-marco-MiniLM-L-6-v2), Anthropic Claude (contextualization), pypdf

## Run It
Open in Colab, add `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` to Secrets, run top to bottom.