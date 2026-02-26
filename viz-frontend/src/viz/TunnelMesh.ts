import * as THREE from 'three';
import { LANE_WIDTH, TUNNEL_LENGTH_METERS, TUBE_SEPARATION, laneCenterY } from './mapping';

export interface TunnelParts {
  group: THREE.Group;
  shellGroup: THREE.Group;
  equipmentGroup: THREE.Group;
  laneMarkings: THREE.Group;
}

function makeTunnelProfile(width = 11, height = 7.5, archRadius = 3.5): THREE.Shape {
  const shape = new THREE.Shape();
  const half = width / 2;

  shape.moveTo(-half, 0);
  shape.lineTo(-half, height - archRadius);
  shape.absarc(0, height - archRadius, half, Math.PI, 0, false);
  shape.lineTo(half, 0);
  shape.lineTo(-half, 0);
  return shape;
}

function makeTunnelShellGeometry(shape: THREE.Shape, length: number, wallThickness: number): THREE.ExtrudeGeometry {
  const outer = shape.clone();
  const hole = makeTunnelProfile(11 - wallThickness * 2, 7.5 - wallThickness * 0.8, 3.1);
  outer.holes = [hole];
  return new THREE.ExtrudeGeometry(outer, {
    steps: 220,
    bevelEnabled: false,
    extrudePath: new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -length),
    ]),
  });
}

export function createTunnelMesh(): TunnelParts {
  const group = new THREE.Group();
  const shellGroup = new THREE.Group();
  const equipmentGroup = new THREE.Group();
  const laneMarkings = new THREE.Group();

  const shellThickness = 0.6;
  const profile = makeTunnelProfile(11, 7.5, 3.5);
  const shellGeometry = makeTunnelShellGeometry(profile, TUNNEL_LENGTH_METERS, shellThickness);

  const shellMaterial = new THREE.MeshStandardMaterial({
    color: 0x54698f,
    roughness: 0.85,
    metalness: 0.05,
    transparent: true,
    opacity: 0.26,
    depthWrite: false,
    side: THREE.DoubleSide,
    clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, -1, 0), 4.8)],
  });
  const liningMaterial = new THREE.MeshStandardMaterial({
    color: 0x263452,
    roughness: 0.92,
    metalness: 0.03,
    transparent: true,
    opacity: 0.2,
    depthWrite: false,
    side: THREE.DoubleSide,
    clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, -1, 0), 4.8)],
  });
  const roadMaterial = new THREE.MeshStandardMaterial({ color: 0x2f3645, roughness: 0.95, metalness: 0.05 });
  const markMaterial = new THREE.MeshBasicMaterial({ color: 0xe5edf8, transparent: true, opacity: 0.95 });

  const nicheMaterial = new THREE.MeshStandardMaterial({ color: 0x344a6a, roughness: 0.8, metalness: 0.05 });
  const cameraMaterial = new THREE.MeshStandardMaterial({ color: 0xd7e0f0, roughness: 0.25, metalness: 0.55 });
  const fanMaterial = new THREE.MeshStandardMaterial({ color: 0x7f8ea8, roughness: 0.5, metalness: 0.5 });
  const hydrantMaterial = new THREE.MeshStandardMaterial({ color: 0xe53f3f, roughness: 0.45, metalness: 0.15 });
  const crossPassageMaterial = new THREE.MeshStandardMaterial({ color: 0x6c7691, roughness: 0.7, metalness: 0.08 });
  const doorMaterial = new THREE.MeshStandardMaterial({ color: 0xaeb6c7, roughness: 0.5, metalness: 0.35 });

  for (const tube of [1, 2]) {
    const tubeCenterX = tube === 1 ? -TUBE_SEPARATION / 2 : TUBE_SEPARATION / 2;

    const outerShell = new THREE.Mesh(shellGeometry, shellMaterial);
    outerShell.rotation.y = Math.PI;
    outerShell.position.set(tubeCenterX, 0, TUNNEL_LENGTH_METERS);
    shellGroup.add(outerShell);

    const innerLining = new THREE.Mesh(
      new THREE.ExtrudeGeometry(profile, {
        steps: 220,
        bevelEnabled: false,
        extrudePath: new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0, 0),
          new THREE.Vector3(0, 0, -TUNNEL_LENGTH_METERS),
        ]),
      }),
      liningMaterial,
    );
    innerLining.rotation.y = Math.PI;
    innerLining.position.copy(outerShell.position);
    shellGroup.add(innerLining);

    const road = new THREE.Mesh(new THREE.PlaneGeometry(2 * LANE_WIDTH + 3.2, TUNNEL_LENGTH_METERS), roadMaterial);
    road.rotation.x = -Math.PI / 2;
    road.position.set(tubeCenterX, 0.01, TUNNEL_LENGTH_METERS / 2);
    equipmentGroup.add(road);

    for (const lane of [1, 2]) {
      const laneCenterX = laneCenterY(tube, lane);
      const marker = new THREE.Mesh(new THREE.PlaneGeometry(0.16, TUNNEL_LENGTH_METERS), markMaterial);
      marker.rotation.x = -Math.PI / 2;
      marker.position.set(laneCenterX, 0.03, TUNNEL_LENGTH_METERS / 2);
      laneMarkings.add(marker);
    }

    for (let z = 60; z < TUNNEL_LENGTH_METERS; z += 80) {
      const niche = new THREE.Mesh(new THREE.BoxGeometry(2.4, 2.1, 2.2), nicheMaterial);
      niche.position.set(tubeCenterX + LANE_WIDTH * 1.45, 1.5, z);
      equipmentGroup.add(niche);

      const hydrant = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.45, 1.2), hydrantMaterial);
      hydrant.position.set(niche.position.x, 1.0, z);
      equipmentGroup.add(hydrant);
    }

    for (let z = 90; z < TUNNEL_LENGTH_METERS; z += 120) {
      const camera = new THREE.Mesh(new THREE.SphereGeometry(0.35, 12, 12), cameraMaterial);
      camera.position.set(tubeCenterX + LANE_WIDTH * 1.05, 5.6, z);
      equipmentGroup.add(camera);
    }

    for (let z = 110; z < TUNNEL_LENGTH_METERS; z += 160) {
      const fan = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, 3.8, 12), fanMaterial);
      fan.rotation.z = Math.PI / 2;
      fan.position.set(tubeCenterX, 6.1, z);
      equipmentGroup.add(fan);
    }

    for (let z = 40; z < TUNNEL_LENGTH_METERS; z += 12) {
      const light = new THREE.Mesh(
        new THREE.BoxGeometry(0.15, 0.05, 2.2),
        new THREE.MeshStandardMaterial({ color: 0xdde8ff, emissive: 0x9bb8ff, emissiveIntensity: 0.55, roughness: 0.5, metalness: 0.1 }),
      );
      light.position.set(tubeCenterX, 6.35, z);
      equipmentGroup.add(light);
    }
  }

  for (let z = 180; z < TUNNEL_LENGTH_METERS; z += 300) {
    const passage = new THREE.Mesh(new THREE.BoxGeometry(TUBE_SEPARATION - 2.6, 3.2, 7.0), crossPassageMaterial);
    passage.position.set(0, 1.8, z);
    equipmentGroup.add(passage);

    const doorA = new THREE.Mesh(new THREE.BoxGeometry(0.2, 2.2, 1.2), doorMaterial);
    doorA.position.set(-TUBE_SEPARATION / 2 + 1.3, 1.4, z);
    const doorB = doorA.clone();
    doorB.position.x = TUBE_SEPARATION / 2 - 1.3;
    equipmentGroup.add(doorA, doorB);
  }

  group.add(shellGroup, equipmentGroup, laneMarkings);
  return { group, shellGroup, equipmentGroup, laneMarkings };
}
