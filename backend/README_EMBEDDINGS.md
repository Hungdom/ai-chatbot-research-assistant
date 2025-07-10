# Fixing 0% Similarity Issue - Embedding Generation Guide

## Problem
The similarity search was returning 0% because the papers in the database don't have embeddings generated for them. When the system tries to calculate cosine similarity between query embeddings and NULL/empty paper embeddings, it results in 0% similarity.

## Solution
This guide explains how to generate embeddings for existing papers in your database.

## Prerequisites
1. OpenAI API key configured in your `.env` file
2. PostgreSQL database with pgvector extension installed
3. Python environment with required dependencies

## Steps to Fix

### 1. Check Your Environment
Make sure your `.env` file contains:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Run the Embedding Generation Script
```bash
cd backend
python generate_embeddings.py
```

### 3. Advanced Usage
You can customize the batch size and delay:
```bash
# Process 5 papers at a time with 2-second delays
python generate_embeddings.py 5 2.0

# Process 20 papers at a time with 0.5-second delays
python generate_embeddings.py 20 0.5
```

### 4. Monitor Progress
The script will show progress like:
```
2024-01-01 10:00:00 - __main__ - INFO - Found 1000 papers without embeddings
2024-01-01 10:00:01 - __main__ - INFO - Processing batch 1 (papers 1-10)
2024-01-01 10:00:02 - __main__ - INFO - ✓ Updated embedding for paper 123
2024-01-01 10:00:03 - __main__ - INFO - Batch 1 complete. Progress: 10/1000 (1.0%)
```

## What the Script Does

1. **Identifies papers without embeddings**: Counts papers where `embedding IS NULL`
2. **Processes in batches**: Avoids overwhelming the OpenAI API
3. **Generates embeddings**: Combines title and abstract for each paper
4. **Updates database**: Stores embeddings in proper vector format
5. **Handles rate limiting**: Adds delays between API calls
6. **Provides progress tracking**: Shows completion percentage

## After Running the Script

Once embeddings are generated, the similarity search will work properly:
- Query embeddings will be compared against paper embeddings
- Similarity scores will be meaningful (e.g., 0.85 = 85% similarity)
- Papers will be ranked by relevance to your query

## Fallback Behavior

The improved system now includes fallback behavior:
- If no embeddings are available, it falls back to keyword search
- If embedding generation fails, it uses text-based search
- This ensures the system always returns relevant results

## Cost Estimation

OpenAI embedding costs (as of 2024):
- ~$0.0001 per 1K tokens
- Average paper (title + abstract) ≈ 200-300 tokens
- 1000 papers ≈ $0.02-0.03

## Troubleshooting

### Common Issues:

1. **"OpenAI API key not configured"**
   - Check your `.env` file
   - Restart your application after adding the key

2. **Rate limiting errors**
   - Increase the delay parameter: `python generate_embeddings.py 10 2.0`
   - Reduce batch size: `python generate_embeddings.py 5 1.0`

3. **Database connection errors**
   - Ensure PostgreSQL is running
   - Check database credentials in `.env`

4. **Out of memory errors**
   - Reduce batch size to 5 or less
   - The script processes one paper at a time within each batch

### Verification Commands:

Check how many papers have embeddings:
```sql
SELECT COUNT(*) FROM arxiv WHERE embedding IS NOT NULL;
```

Check total papers:
```sql
SELECT COUNT(*) FROM arxiv;
```

Test cosine similarity:
```sql
SELECT title, cosine_similarity(embedding, embedding) as self_similarity 
FROM arxiv 
WHERE embedding IS NOT NULL 
LIMIT 5;
```

## Performance Tips

1. **Optimal batch size**: 10-20 papers per batch
2. **Delay settings**: 1-2 seconds between requests
3. **Run during off-peak hours**: Avoid API rate limits
4. **Monitor API usage**: Check OpenAI dashboard for usage

## Future Maintenance

- Run the script when new papers are added to the database
- Consider setting up automatic embedding generation for new papers
- Monitor embedding quality and update if needed

## System Improvements Made

1. **Better error handling**: Graceful fallbacks when embeddings fail
2. **Format fixes**: Proper vector format for PostgreSQL
3. **Similarity thresholding**: Only show papers with >10% similarity
4. **Keyword fallback**: Always return relevant results
5. **Comprehensive logging**: Better debugging and monitoring 