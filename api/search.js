// api/search.js
import { Configuration, OpenAIApi } from 'openai';
import { PineconeClient } from '@pinecone-database/pinecone';

const openai = new OpenAIApi(
  new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
  })
);

const pinecone = new PineconeClient();

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Only POST requests allowed' });
  }

  const { query } = req.body;

  try {
    await pinecone.init({
      apiKey: process.env.PINECONE_API_KEY,
      environment: process.env.PINECONE_ENV,
    });

    const embed = await openai.createEmbedding({
      model: 'text-embedding-ada-002',
      input: query,
    });

    const [{ embedding }] = embed.data.data;

    const index = pinecone.Index(process.env.PINECONE_INDEX);

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

