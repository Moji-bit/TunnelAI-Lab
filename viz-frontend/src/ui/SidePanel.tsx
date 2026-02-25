import type { Selection } from '../state/store';

interface SidePanelProps {
  selection: Selection;
}

export function SidePanel({ selection }: SidePanelProps): JSX.Element {
  return (
    <aside className="side-panel">
      <div className="panel-box">
        <strong>Selection</strong>
      </div>
      {!selection && <div className="panel-box">Click a vehicle or event.</div>}
      {selection?.kind === 'vehicle' && (
        <div className="panel-box">
          <div>ID: {selection.data.id}</div>
          <div>tube: {selection.data.tube}</div>
          <div>lane: {selection.data.lane}</div>
          <div>x: {selection.data.x.toFixed(1)}</div>
          <div>v: {selection.data.v.toFixed(1)}</div>
          <div>a: {selection.data.a.toFixed(2)}</div>
          <div>type: {selection.data.type}</div>
        </div>
      )}
      {selection?.kind === 'event' && (
        <div className="panel-box">
          <div>ID: {selection.data.id}</div>
          <div>type: {selection.data.type}</div>
          <div>tube: {selection.data.tube}</div>
          <div>lane: {selection.data.lane}</div>
          <div>x0/x1: {selection.data.x0.toFixed(1)} / {selection.data.x1.toFixed(1)}</div>
          <div>active: {String(selection.data.active)}</div>
          <div>severity: {selection.data.severity ?? 0}</div>
        </div>
      )}
    </aside>
  );
}
