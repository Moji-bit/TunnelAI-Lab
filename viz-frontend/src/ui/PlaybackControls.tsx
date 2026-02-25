interface PlaybackControlsProps {
  paused: boolean;
  speed: number;
  disabled: boolean;
  onPlayPause: () => void;
  onSpeedChange: (speed: number) => void;
}

const SPEEDS = [0.5, 1, 2, 5, 10, 20];

export function PlaybackControls(props: PlaybackControlsProps): JSX.Element {
  return (
    <div className="playback-controls">
      <button onClick={props.onPlayPause} disabled={props.disabled}>
        {props.paused ? 'Play' : 'Pause'}
      </button>
      <label>
        Speed{' '}
        <select
          value={String(props.speed)}
          onChange={(event) => props.onSpeedChange(Number(event.target.value))}
          disabled={props.disabled}
        >
          {SPEEDS.map((speed) => (
            <option key={speed} value={speed}>
              {speed}x
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
