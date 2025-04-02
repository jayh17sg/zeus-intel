import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENV,
});

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Only POST requests allowed' });
  }

  const { query } = req.body;

  try {
    const embed = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: query,
    });

    const [{ embedding }] = embed.data;

    const index = pinecone.index(process.env.PINECONE_INDEX);

    const results = await index.query({
      vector: embedding,
      topK: 5,
      includeMetadata: true,
    });

    return res.status(200).json({ results: results.matches });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: error.message });
  }
}
