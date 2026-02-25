# SCADA Hybrid Dynamics Model

This model upgrades TunnelAI-Lab from simple random traces to a coupled, hybrid SCADA-style simulator.

## Coupled causal chain
`Traffic -> Emission -> Air quality -> Ventilation -> Air quality feedback`

## What is now modeled

1. **Traffic nonlinearity + queue pressure**
   - Fundamental diagram speed relation
   - Capacity-shaped outflow (saturation around critical density)
   - Queue index that increases when inflow exceeds capacity and drags speed

2. **Emission inertia and nonlinearity**
   - CO source depends on density, speed, heavy-vehicle ratio, incident severity
   - Slow first-order source lag (`source_tau_s`) to create delayed CO behavior
   - Ventilation removal term coupled to fan stage

3. **Discrete actuator behavior (fan stages)**
   - Fan stage control with hysteresis
   - Stage transitions 0..3 (no continuous actuator values)
   - Uses CO and VIS proxy as control signal

4. **Weather and incident interactions**
   - Weather can lower visibility and speed limits
   - Incident severity amplifies emission and visibility degradation

## Key files
- `sim/traffic_model.py`
- `sim/emission_model.py`
- `streaming/opcua_mock_server.py`

## Why this is "hybrid"
- Physics-inspired continuous dynamics (traffic/emissions/ventilation)
- Heuristic SCADA control logic (discrete fan stages + hysteresis)
- This is suitable for ML research where realism + controllability are both needed.
