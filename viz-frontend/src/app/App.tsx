import { useEffect, useMemo, useRef, useState } from 'react';
import { fetchScenarios } from '../api/scenarios';
import type { EventState, Frame, LayerVisibility, Mode, ScenarioSummary, Selection, VehicleState } from '../state/store';
import { LayerToggles } from '../ui/LayerToggles';
import { PlaybackControls } from '../ui/PlaybackControls';
import { SidePanel } from '../ui/SidePanel';
import { TopBar } from '../ui/TopBar';
import { ThreeScene } from '../viz/ThreeScene';
import { TunnelWsClient, type WsStats } from '../ws/wsClient';

const DEFAULT_SCENARIO = 'stau_case_00';
const BUFFER_SECONDS = 2;

function interpolateVehicles(frame0: Frame, frame1: Frame, alpha: number): VehicleState[] {
  const nextById = new Map(frame1.vehicles.map((vehicle) => [vehicle.id, vehicle]));
  return frame0.vehicles.map((vehicle0) => {
    const vehicle1 = nextById.get(vehicle0.id);
    if (!vehicle1) {
      return vehicle0;
    }
    return {
      ...vehicle0,
      x: vehicle0.x + (vehicle1.x - vehicle0.x) * alpha,
      v: vehicle0.v + (vehicle1.v - vehicle0.v) * alpha,
      a: vehicle0.a + (vehicle1.a - vehicle0.a) * alpha,
      lane: alpha > 0.5 ? vehicle1.lane : vehicle0.lane,
      tube: alpha > 0.5 ? vehicle1.tube : vehicle0.tube,
    };
  });
}

function interpolateFrame(frame0: Frame, frame1: Frame, alpha: number): { vehicles: VehicleState[]; events: EventState[]; t: number } {
  return {
    vehicles: interpolateVehicles(frame0, frame1, alpha),
    events: frame1.events,
    t: frame0.t + (frame1.t - frame0.t) * alpha,
  };
}

export function App(): JSX.Element {
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [scenarioId, setScenarioId] = useState(DEFAULT_SCENARIO);
  const [mode, setMode] = useState<Mode>('playback');
  const [stats, setStats] = useState<WsStats>({ status: 'disconnected', latencyMs: 0 });
  const [fps, setFps] = useState(0);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [layers, setLayers] = useState<LayerVisibility>({ vehicles: true, events: true, laneMarkings: true, debugHud: true });
  const [selection, setSelection] = useState<Selection>(null);
  const [hud, setHud] = useState({ t: 0, vehicles: 0 });

  const canvasRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<ThreeScene | null>(null);
  const wsRef = useRef<TunnelWsClient | null>(null);
  const bufferRef = useRef<Frame[]>([]);
  const speedRef = useRef(speed);
  const pausedRef = useRef(paused);

  useEffect(() => {
    fetchScenarios()
      .then((response) => {
        setScenarios(response);
        if (response.length > 0) {
          setScenarioId(response.some((scenario) => scenario.id === DEFAULT_SCENARIO) ? DEFAULT_SCENARIO : response[0].id);
        }
      })
      .catch(() => setScenarios([{ id: DEFAULT_SCENARIO, name: DEFAULT_SCENARIO, duration: 0, dt: 1, tubes: 2, lanes: 2, tags: [] }]));
  }, []);

  useEffect(() => {
    if (!canvasRef.current) {
      return;
    }

    const scene = new ThreeScene(
      canvasRef.current,
      (sceneSelection) => {
        if (!sceneSelection) {
          setSelection(null);
          scene.setSelectedVehicle(null);
          return;
        }

        const latest = bufferRef.current.length > 0 ? bufferRef.current[bufferRef.current.length - 1] : undefined;
        if (!latest) {
          return;
        }

        if (sceneSelection.vehicleId) {
          const found = latest.vehicles.find((vehicle: VehicleState) => vehicle.id === sceneSelection.vehicleId);
          if (found) {
            setSelection({ kind: 'vehicle', data: found });
            scene.setSelectedVehicle(found.id);
          }
        }

        if (sceneSelection.eventId) {
          const found = latest.events.find((event: EventState) => event.id === sceneSelection.eventId);
          if (found) {
            setSelection({ kind: 'event', data: found });
            scene.setSelectedVehicle(null);
          }
        }
      },
      setFps,
    );

    sceneRef.current = scene;
    scene.setLayers(layers);

    const client = new TunnelWsClient();
    wsRef.current = client;
    client.setStatsHandler(setStats);
    client.setFrameHandler((frame) => {
      const local = bufferRef.current;
      local.push(frame);
      const frameLimit = Math.max(2, Math.ceil(BUFFER_SECONDS / Math.max(0.05, frame.dt)));
      while (local.length > frameLimit) {
        local.shift();
      }
    });

    let handle = 0;
    const loop = () => {
      handle = requestAnimationFrame(loop);
      const buffer = bufferRef.current;
      if (buffer.length >= 2 && sceneRef.current) {
        if (pausedRef.current) {
          return;
        }
        const newest = buffer[buffer.length - 1];
        const older = buffer[buffer.length - 2];
                const simulatedT = newest.t - (120 / 1000) * speedRef.current;

        let frame0 = older;
        let frame1 = newest;
        for (let i = 0; i < buffer.length - 1; i += 1) {
          if (buffer[i].t <= simulatedT && buffer[i + 1].t >= simulatedT) {
            frame0 = buffer[i];
            frame1 = buffer[i + 1];
            break;
          }
        }

        const alpha = Math.min(1, Math.max(0, (simulatedT - frame0.t) / Math.max(0.001, frame1.t - frame0.t)));
        const interpolated = interpolateFrame(frame0, frame1, alpha);
        sceneRef.current.renderFrame(interpolated.vehicles, interpolated.events);

        setHud({ t: interpolated.t, vehicles: interpolated.vehicles.length });
      }
    };
    loop();

    return () => {
      cancelAnimationFrame(handle);
      client.disconnect();
      scene.dispose();
      sceneRef.current = null;
      wsRef.current = null;
    };
  }, []);

  useEffect(() => {
    sceneRef.current?.setLayers(layers);
  }, [layers]);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  const playbackDisabled = useMemo(() => mode !== 'playback' || stats.status !== 'connected', [mode, stats.status]);

  const connect = () => {
    const params = new URLSearchParams(window.location.search);
    const sessionId = params.get('session_id') ?? undefined;
    wsRef.current?.connect(mode, scenarioId, sessionId);
    bufferRef.current = [];
    setSelection(null);
  };

  const disconnect = () => {
    wsRef.current?.disconnect();
    bufferRef.current = [];
    setSelection(null);
  };

  const togglePlayPause = () => {
    const nextPaused = !paused;
    setPaused(nextPaused);
    wsRef.current?.sendControl({ cmd: nextPaused ? 'pause' : 'play' });
  };

  const changeSpeed = (nextSpeed: number) => {
    setSpeed(nextSpeed);
    wsRef.current?.sendControl({ cmd: 'speed', factor: nextSpeed });
  };

  return (
    <div className="app-layout">
      <TopBar
        scenarios={scenarios}
        selectedScenario={scenarioId}
        mode={mode}
        stats={stats}
        fps={fps}
        onScenarioChange={setScenarioId}
        onModeChange={setMode}
        onConnect={connect}
        onDisconnect={disconnect}
      />

      <PlaybackControls paused={paused} speed={speed} disabled={playbackDisabled} onPlayPause={togglePlayPause} onSpeedChange={changeSpeed} />
      <LayerToggles layers={layers} onChange={setLayers} />

      <div className="viewer-row">
        <div className="viewer" ref={canvasRef}>
          {layers.debugHud && (
            <div className="hud">
              <div>ws: {stats.status}</div>
              <div>t: {hud.t.toFixed(2)}</div>
              <div>#vehicles: {hud.vehicles}</div>
              <div>fps: {fps.toFixed(0)}</div>
            </div>
          )}
        </div>
        <SidePanel selection={selection} />
      </div>
    </div>
  );
}
