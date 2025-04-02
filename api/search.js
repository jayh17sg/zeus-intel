import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';

// OpenAI initialization
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Pinecone initialization
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

export default async function handler(req, res) {
  // ✅ CORS HEADERS
  res.setHeader('Access-Control-Allow-Origin', '*'); // You can restrict to a specific domain later
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // ✅ Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Only POST requests allowed' });
  }

  const { query } = req.body;

  try {
    // Step 1: Embed the query using OpenAI
    const embed = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: query,
    });

    const [{ embedding }] = embed.data;

    // Step 2: Query Pinecone
    const index = pinecone.index(process.env.PINECONE_INDEX);

    const results = await index.query({
      vector: embedding,
      topK: 5,
      includeMetadata: true,
    });

    // Step 3: Return results
    return res.status(200).json({ results: results.matches });
  } catch (error) {
    console.error('🔥 Error in /api/search:', error);
    return res.status(500).json({ error: error.message });
  }
}
