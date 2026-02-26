import type { LayerVisibility } from '../state/store';

interface LayerTogglesProps {
  layers: LayerVisibility;
  onChange: (layers: LayerVisibility) => void;
}

export function LayerToggles({ layers, onChange }: LayerTogglesProps): JSX.Element {
  const update = (key: keyof LayerVisibility, value: boolean) => onChange({ ...layers, [key]: value });

  return (
    <div className="layer-toggles">
      {(Object.keys(layers) as (keyof LayerVisibility)[]).map((key) => (
        <label key={key}>
          <input type="checkbox" checked={layers[key]} onChange={(event) => update(key, event.target.checked)} /> {key}
        </label>
      ))}
    </div>
  );
}
