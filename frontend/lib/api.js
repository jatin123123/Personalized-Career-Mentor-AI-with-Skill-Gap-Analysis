const LOCAL_API_URL = 'http://localhost:8001';
const PRODUCTION_API_URL = 'https://career-mentor-api-jatin.azurewebsites.net';

// Vercel injects NEXT_PUBLIC_API_URL at build time. Keep a production fallback
// so a missing Vercel variable cannot silently point the deployed app at the
// visitor's own localhost.
export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (process.env.NODE_ENV === 'production' ? PRODUCTION_API_URL : LOCAL_API_URL);
