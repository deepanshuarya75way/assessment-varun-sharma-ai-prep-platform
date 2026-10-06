const fs = require("fs");
const path = require("path");
const { getRedis } = require("../config/redis");
const { embedText } = require("../services/llm.service");

const knowledge = JSON.parse(fs.readFileSync(path.join(__dirname, "knowledge", "interview-kb.json"), "utf8"));
let localEmbeddings = new Map();
let initialized = false;

function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return aa && bb ? dot / (Math.sqrt(aa) * Math.sqrt(bb)) : 0;
}

function lexicalScore(query, text) {
  const q = new Set(query.toLowerCase().match(/[a-z0-9+#.]+/g) || []);
  const t = new Set(text.toLowerCase().match(/[a-z0-9+#.]+/g) || []);
  let hits = 0; for (const token of q) if (t.has(token)) hits++;
  return q.size ? hits / q.size : 0;
}

async function ensureInitialized() {
  if (initialized) return;
  const redis = getRedis();
  for (const doc of knowledge) {
    const key = `ai-prep:rag:embedding:${doc.id}`;
    let embedding = null;
    if (redis) {
      const cached = await redis.get(key);
      if (cached) embedding = JSON.parse(cached);
    }
    if (!embedding) embedding = await embedText(`${doc.topic}: ${doc.text}`);
    if (embedding) {
      localEmbeddings.set(doc.id, embedding);
      if (redis) await redis.set(key, JSON.stringify(embedding), { EX: 60 * 60 * 24 * 30 });
    }
  }
  initialized = true;
}

async function retrieveContext({ resume, jobDescription, selfDescription, topK = 4 }) {
  await ensureInitialized();
  const query = `${jobDescription}\n${selfDescription}\n${resume}`.slice(0, 18000);
  const redis = getRedis();
  const cacheKey = `ai-prep:rag:context:${require("crypto").createHash("sha256").update(query).digest("hex").slice(0, 24)}`;
  if (redis) {
    const cached = await redis.get(cacheKey);
    if (cached) return JSON.parse(cached);
  }

  const scored = knowledge.map((doc) => {
    const semantic = 0;
    return { doc, score: semantic + lexicalScore(query, `${doc.topic} ${doc.text}`) * 0.35 };
  });

  // If an embedding for the query is available, use it for the semantic part.
  const queryEmbedding = await embedText(query);
  if (queryEmbedding) for (const item of scored) item.score = cosine(queryEmbedding, localEmbeddings.get(item.doc.id)) * 0.65 + lexicalScore(query, `${item.doc.topic} ${item.doc.text}`) * 0.35;

  const candidates = scored.sort((a,b) => b.score - a.score).slice(0, Math.max(Number(topK) * 5, 20));
  let result;

  try{
    const reranked = await rerankDocuments(query, candidates);

    result = reranked.slice(0, Number(topK)).map(({doc}) => ({
      topic: doc.topic,
      text: doc.text,
    }));
  }

  catch(error) {
    console.warn("Rerank failed");

    result = candidates.slice(0, Number(topK)).map(({doc}) => ({
      topic: doc.topic,
      text: doc.text,
    }));
  }


  return result;

  // const result = scored.sort((a, b) => b.score - a.score).slice(0, Number(topK)).map(({ doc }) => `[${doc.topic}] ${doc.text}`);
  // if (redis) await redis.set(cacheKey, JSON.stringify(result), { EX: 1800 });
  // return result;
}

module.exports = { retrieveContext };
