import * as THREE from 'three';
import { LANE_WIDTH, TUNNEL_LENGTH_METERS, TUBE_SEPARATION, laneCenterY } from './mapping';

export interface TunnelParts {
  group: THREE.Group;
  shellGroup: THREE.Group;
  equipmentGroup: THREE.Group;
}

export function createTunnelMesh(): TunnelParts {
  const group = new THREE.Group();
  const shellGroup = new THREE.Group();
  const equipmentGroup = new THREE.Group();

  const tunnelRadius = LANE_WIDTH * 2.35;
  const shellThickness = 0.7;

  const shellMaterial = new THREE.MeshStandardMaterial({
    color: 0x54698f,
    roughness: 0.85,
    metalness: 0.05,
    transparent: true,
    opacity: 0.26,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const liningMaterial = new THREE.MeshStandardMaterial({
    color: 0x263452,
    roughness: 0.92,
    metalness: 0.03,
    transparent: true,
    opacity: 0.2,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const roadMaterial = new THREE.MeshStandardMaterial({ color: 0x2f3645, roughness: 0.95, metalness: 0.05 });
  const markMaterial = new THREE.MeshBasicMaterial({ color: 0xe5edf8 });
  const ventilationMaterial = new THREE.MeshStandardMaterial({ color: 0x7f8ea8, roughness: 0.5, metalness: 0.5 });
  const cameraMaterial = new THREE.MeshStandardMaterial({ color: 0xcfd8e8, roughness: 0.25, metalness: 0.6 });
  const extinguisherMaterial = new THREE.MeshStandardMaterial({ color: 0xe53f3f, roughness: 0.45, metalness: 0.15 });
  const nicheMaterial = new THREE.MeshStandardMaterial({ color: 0x2d405f, roughness: 0.8, metalness: 0.05 });
  const crossPassageMaterial = new THREE.MeshStandardMaterial({ color: 0x69758d, roughness: 0.7, metalness: 0.1 });

  const nicheMaterial = new THREE.MeshStandardMaterial({ color: 0x344a6a, roughness: 0.8, metalness: 0.05 });
  const cameraMaterial = new THREE.MeshStandardMaterial({ color: 0xd7e0f0, roughness: 0.25, metalness: 0.55 });
  const fanMaterial = new THREE.MeshStandardMaterial({ color: 0x7f8ea8, roughness: 0.5, metalness: 0.5 });
  const hydrantMaterial = new THREE.MeshStandardMaterial({ color: 0xe53f3f, roughness: 0.45, metalness: 0.15 });
  const crossPassageMaterial = new THREE.MeshStandardMaterial({ color: 0x6c7691, roughness: 0.7, metalness: 0.08 });
  const doorMaterial = new THREE.MeshStandardMaterial({ color: 0xaeb6c7, roughness: 0.5, metalness: 0.35 });

  for (const tube of [1, 2]) {
    const road = new THREE.Mesh(new THREE.PlaneGeometry(TUNNEL_LENGTH_METERS, 2 * LANE_WIDTH + 3.2), roadMaterial);
    road.rotation.x = -Math.PI / 2;
    road.position.set(tubeCenterX, 0.01, TUNNEL_LENGTH_METERS / 2);
    equipmentGroup.add(road);

    for (const lane of [1, 2]) {
      const laneCenter = laneCenterY(tube, lane);
      const marker = new THREE.Mesh(new THREE.PlaneGeometry(TUNNEL_LENGTH_METERS, 0.16), markMaterial);
      marker.rotation.x = -Math.PI / 2;
      marker.position.set(laneCenterX, 0.03, TUNNEL_LENGTH_METERS / 2);
      equipmentGroup.add(marker);
    }

    const tubeCenterY = tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2;
    const outerShell = new THREE.Mesh(
      new THREE.CylinderGeometry(
        tunnelRadius + shellThickness,
        tunnelRadius + shellThickness,
        TUNNEL_LENGTH_METERS,
        56,
        1,
        false,
      ),
      wallMaterial,
    );
    outerShell.rotation.z = Math.PI / 2;
    outerShell.position.set(TUNNEL_LENGTH_METERS / 2, tubeCenterY, tunnelRadius - 0.25);
    group.add(outerShell);

    const rightWall = leftWall.clone();
    rightWall.position.y =
      (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) - (LANE_WIDTH + 0.25);
    group.add(leftWall, rightWall);

    for (let x = 70; x < TUNNEL_LENGTH_METERS; x += 120) {
      const ventilation = new THREE.Mesh(new THREE.BoxGeometry(4.5, 0.8, 0.35), ventilationMaterial);
      ventilation.position.set(x, tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2, 2.9);
      group.add(ventilation);

      const camera = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 16), cameraMaterial);
      camera.position.set(x + 20, (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) + 1.85, 1.65);
      group.add(camera);

      const extinguisher = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.25, 0.95), extinguisherMaterial);
      extinguisher.position.set(x + 35, (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) - 2.05, 0.75);
      group.add(extinguisher);

      const emergencyNiche = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.7, 1.9), nicheMaterial);
      emergencyNiche.position.set(x + 46, (tube === 1 ? TUBE_SEPARATION / 2 : -TUBE_SEPARATION / 2) + 2.08, 0.95);
      group.add(emergencyNiche);
    }
  }

  for (let x = 150; x < TUNNEL_LENGTH_METERS; x += 260) {
    const crossPassage = new THREE.Mesh(new THREE.BoxGeometry(3.0, TUBE_SEPARATION - 0.8, 2.8), crossPassageMaterial);
    crossPassage.position.set(x, 0, 1.4);
    group.add(crossPassage);
  }

  group.add(shellGroup, equipmentGroup);
  return { group, shellGroup, equipmentGroup };
}
