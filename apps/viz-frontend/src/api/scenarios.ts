import type { ScenarioSummary } from '../state/store';

const HTTP_BASE = import.meta.env.VITE_HTTP_BASE ?? 'http://127.0.0.1:8000';

export async function fetchScenarios(): Promise<ScenarioSummary[]> {
  const response = await fetch(`${HTTP_BASE}/api/scenarios`);
  if (!response.ok) {
    throw new Error(`Failed to load scenarios: ${response.status}`);
  }
  return response.json() as Promise<ScenarioSummary[]>;
}
