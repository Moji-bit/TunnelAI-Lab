import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { EventState, LayerVisibility, VehicleState } from '../state/store';
import { TUNNEL_LENGTH_METERS } from './mapping';
import { pointerToNdc } from './Picking';
import { createTunnelMesh } from './TunnelMesh';
import { EventsLayer } from './EventsLayer';
import { VehiclesLayer } from './VehiclesLayer';

export interface SceneSelection {
  vehicleId?: string;
  eventId?: string;
}

export class ThreeScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly camera: THREE.OrthographicCamera;
  private readonly scene: THREE.Scene;
  private readonly controls: OrbitControls;
  private readonly vehiclesLayer = new VehiclesLayer();
  private readonly eventsLayer = new EventsLayer();
  private readonly laneMarkings: THREE.Group;
  private readonly debugHelpers = new THREE.Group();
  private readonly raycaster = new THREE.Raycaster();

  private animationHandle = 0;
  private lastFpsSample = performance.now();
  private frameCount = 0;

  constructor(
    private readonly container: HTMLElement,
    private readonly onSelect: (selection: SceneSelection | null) => void,
    private readonly onFps: (fps: number) => void,
  ) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b1220);

    this.camera = new THREE.OrthographicCamera(-260, 260, 150, -150, 0.1, 5000);
    this.camera.position.set(300, 0, 260);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(new THREE.Vector3(220, 0, 0));

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);

    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableRotate = false;
    this.controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    this.controls.target.set(TUNNEL_LENGTH_METERS * 0.22, 0, 0);
    this.controls.zoomSpeed = 0.9;
    this.controls.minZoom = 0.25;
    this.controls.maxZoom = 8;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const directional = new THREE.DirectionalLight(0xffffff, 0.9);
    directional.position.set(200, 300, 200);
    this.scene.add(directional);

    const secondaryDirectional = new THREE.DirectionalLight(0xaecaff, 0.45);
    secondaryDirectional.position.set(-180, -200, 160);
    this.scene.add(secondaryDirectional);

    this.debugHelpers.add(new THREE.AxesHelper(50));
    this.debugHelpers.add(new THREE.GridHelper(400, 40, 0x375283, 0x253654));
    this.debugHelpers.position.set(0, 0, 0.02);
    this.scene.add(this.debugHelpers);

    const tunnel = createTunnelMesh();
    this.laneMarkings = tunnel.laneMarkings;
    this.scene.add(tunnel.group);
    this.scene.add(this.vehiclesLayer.group);
    this.scene.add(this.eventsLayer.group);

    this.renderer.domElement.addEventListener('pointerdown', this.handlePointerDown);
    window.addEventListener('resize', this.handleResize);
    this.handleResize();

    this.animate();
  }

  dispose(): void {
    cancelAnimationFrame(this.animationHandle);
    this.renderer.domElement.removeEventListener('pointerdown', this.handlePointerDown);
    window.removeEventListener('resize', this.handleResize);
    this.controls.dispose();
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
  }

  renderFrame(vehicles: VehicleState[], events: EventState[]): void {
    this.vehiclesLayer.update(vehicles);
    this.eventsLayer.update(events);
  }

  setLayers(layers: LayerVisibility): void {
    this.vehiclesLayer.setVisible(layers.vehicles);
    this.eventsLayer.setVisible(layers.events);
    this.laneMarkings.visible = layers.laneMarkings;
    this.debugHelpers.visible = layers.debugHud;
  }

  setSelectedVehicle(vehicleId: string | null): void {
    this.vehiclesLayer.setSelectedVehicle(vehicleId);
  }

  private handleResize = (): void => {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    const frustum = 220;
    const aspect = width / Math.max(1, height);
    this.camera.left = -frustum * aspect;
    this.camera.right = frustum * aspect;
    this.camera.top = frustum * 0.55;
    this.camera.bottom = -frustum * 0.55;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  };

  private handlePointerDown = (event: PointerEvent): void => {
    const ndc = pointerToNdc(event, this.renderer.domElement);
    this.raycaster.setFromCamera(ndc, this.camera);
    const intersects = this.raycaster.intersectObjects([this.vehiclesLayer.group, this.eventsLayer.group], true);

    for (const hit of intersects) {
      const vehicleId = this.vehiclesLayer.resolveVehicleFromIntersection(hit);
      if (vehicleId) {
        this.onSelect({ vehicleId });
        return;
      }
      const eventId = this.eventsLayer.resolveEventId(hit.object);
      if (eventId) {
        this.onSelect({ eventId });
        return;
      }
    }
    this.onSelect(null);
  };

  private animate = (): void => {
    this.animationHandle = requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);

    this.frameCount += 1;
    const now = performance.now();
    if (now - this.lastFpsSample >= 1000) {
      this.onFps((this.frameCount * 1000) / (now - this.lastFpsSample));
      this.frameCount = 0;
      this.lastFpsSample = now;
    }
  };
}
