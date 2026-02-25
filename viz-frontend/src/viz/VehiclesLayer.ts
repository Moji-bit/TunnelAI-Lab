import * as THREE from 'three';
import type { VehicleState } from '../state/store';
import { laneCenterY, xToWorld } from './mapping';

const MAX_INSTANCES = 6000;

interface MeshBundle {
  mesh: THREE.InstancedMesh;
  indexToVehicleId: Map<number, string>;
}

export class VehiclesLayer {
  readonly group = new THREE.Group();

  private meshes: Record<VehicleState['type'], MeshBundle>;
  private selectedVehicleId: string | null = null;

  constructor() {
    this.meshes = {
      car: this.createMeshBundle(new THREE.BoxGeometry(4.4, 0.72, 0.4), 0x2b91ff),
      truck: this.createMeshBundle(new THREE.BoxGeometry(8.4, 0.82, 0.45), 0x39d26a),
      emergency: this.createMeshBundle(new THREE.BoxGeometry(5.2, 0.78, 0.45), 0xe43a52),
    };

    this.group.add(this.meshes.car.mesh, this.meshes.truck.mesh, this.meshes.emergency.mesh);
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  setSelectedVehicle(id: string | null): void {
    this.selectedVehicleId = id;
  }

  update(vehicles: VehicleState[]): void {
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    for (const bundle of Object.values(this.meshes)) {
      bundle.mesh.count = 0;
      bundle.indexToVehicleId.clear();
    }

    const counters: Record<VehicleState['type'], number> = { car: 0, truck: 0, emergency: 0 };

    vehicles.forEach((vehicle) => {
      const bundle = this.meshes[vehicle.type];
      const idx = counters[vehicle.type]++;
      dummy.position.set(xToWorld(vehicle.x), laneCenterY(vehicle.tube, vehicle.lane), 0.25);
      dummy.updateMatrix();
      bundle.mesh.setMatrixAt(idx, dummy.matrix);
      color.set(vehicle.id === this.selectedVehicleId ? 0xffcf40 : 0xffffff);
      bundle.mesh.setColorAt(idx, color);
      bundle.indexToVehicleId.set(idx, vehicle.id);
    });

    Object.entries(this.meshes).forEach(([type, bundle]) => {
      const activeCount = counters[type as VehicleState['type']];
      bundle.mesh.count = activeCount;
      bundle.mesh.instanceMatrix.needsUpdate = true;
      if (bundle.mesh.instanceColor) {
        bundle.mesh.instanceColor.needsUpdate = true;
      }
    });
  }

  resolveVehicleFromIntersection(intersection: THREE.Intersection<THREE.Object3D<THREE.Object3DEventMap>>): string | null {
    const instanceId = intersection.instanceId;
    if (instanceId == null) {
      return null;
    }
    for (const bundle of Object.values(this.meshes)) {
      if (intersection.object === bundle.mesh) {
        return bundle.indexToVehicleId.get(instanceId) ?? null;
      }
    }
    return null;
  }

  private createMeshBundle(geometry: THREE.BoxGeometry, colorHex: number): MeshBundle {
    const material = new THREE.MeshStandardMaterial({ color: colorHex, roughness: 0.4, metalness: 0.15 });
    const mesh = new THREE.InstancedMesh(geometry, material, MAX_INSTANCES);
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.frustumCulled = false;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    return { mesh, indexToVehicleId: new Map() };
  }
}
