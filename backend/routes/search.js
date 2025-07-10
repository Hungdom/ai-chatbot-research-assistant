const express = require('express');
const router = express.Router();
const pool = require('../db');

router.post('/', async (req, res) => {
  try {
    const { year, keywords } = req.body;
    
    // Build the query
    let query = `
      SELECT 
        id,
        title,
        authors,
        abstract,
        categories,
        published_date,
        updated_date,
        pdf_url,
        arxiv_url
      FROM arxiv
      WHERE 1=1
    `;
    
    const queryParams = [];
    
    // Add year filter if provided
    if (year) {
      query += ` AND EXTRACT(YEAR FROM published_date) = $${queryParams.length + 1}`;
      queryParams.push(year);
    }
    
    // Add keyword filter if provided
    if (keywords && keywords.length > 0) {
      const keywordConditions = keywords.map((_, index) => {
        const paramIndex = queryParams.length + 1;
        queryParams.push(`%${keywords[index].toLowerCase()}%`);
        return `(LOWER(title) LIKE $${paramIndex} OR LOWER(abstract) LIKE $${paramIndex})`;
      });
      query += ` AND (${keywordConditions.join(' OR ')})`;
    }
    
    // Add ordering and limit
    query += ` ORDER BY published_date DESC LIMIT 50`;
    
    // Execute the query
    const result = await pool.query(query, queryParams);
    
    // Generate a summary if we have results
    let summary = '';
    if (result.rows.length > 0) {
      const yearText = year ? `in ${year}` : '';
      const keywordText = keywords && keywords.length > 0 ? `related to ${keywords.join(', ')}` : '';
      summary = `Found ${result.rows.length} papers ${yearText} ${keywordText}.`;
    }
    
    res.json({
      papers: result.rows,
      summary
    });
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({
      message: 'An error occurred while searching for papers'
    });
  }
});

module.exports = router; 