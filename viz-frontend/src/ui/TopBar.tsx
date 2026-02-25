import type { Mode, ScenarioSummary } from '../state/store';
import type { WsStats } from '../ws/wsClient';

interface TopBarProps {
  scenarios: ScenarioSummary[];
  selectedScenario: string;
  mode: Mode;
  stats: WsStats;
  fps: number;
  onScenarioChange: (scenarioId: string) => void;
  onModeChange: (mode: Mode) => void;
  onConnect: () => void;
  onDisconnect: () => void;
}

export function TopBar(props: TopBarProps): JSX.Element {
  return (
    <div className="topbar">
      <label>
        Scenario{' '}
        <select value={props.selectedScenario} onChange={(event) => props.onScenarioChange(event.target.value)}>
          {props.scenarios.map((scenario) => (
            <option key={scenario.id} value={scenario.id}>
              {scenario.id}
            </option>
          ))}
        </select>
      </label>

      <label>
        Mode{' '}
        <select value={props.mode} onChange={(event) => props.onModeChange(event.target.value as Mode)}>
          <option value="playback">Playback</option>
          <option value="live">Live</option>
        </select>
      </label>

      <button onClick={props.onConnect}>Connect</button>
      <button onClick={props.onDisconnect}>Disconnect</button>

      <span className={`tag ${props.stats.status === 'connected' ? 'ok' : 'warn'}`}>
        {props.stats.status}
      </span>
      <span>latency {Math.round(props.stats.latencyMs)} ms</span>
      <span>fps {props.fps.toFixed(0)}</span>
    </div>
  );
}
