import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

// Types
interface ProcessingStats {
  inferenceTimeMs: number;
  projectionTimeMs: number;
  downsamplingTimeMs: number;
  totalTimeMs: number;
}

interface PointCloudResult {
  plyData: Uint8Array;
  numPoints: number;
  stats: ProcessingStats;
}

// Global state
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let renderer: THREE.WebGLRenderer;
let controls: OrbitControls;
let pointCloud: THREE.Points | null = null;
let currentPlyData: Uint8Array | null = null;
let frameCount = 0;
let lastTime = performance.now();

// Initialize Three.js scene
function initScene(): void {
  const container = document.getElementById('canvas-container')!;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  // Camera
  camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.01,
    1000
  );
  camera.position.set(0, 0, 2);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = true;

  // Grid helper
  const gridHelper = new THREE.GridHelper(10, 20, 0x444444, 0x222222);
  gridHelper.rotation.x = Math.PI / 2;
  scene.add(gridHelper);

  // Axes helper
  const axesHelper = new THREE.AxesHelper(1);
  scene.add(axesHelper);

  // Ambient light
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);

  // Handle resize
  window.addEventListener('resize', onWindowResize);

  // Start animation loop
  animate();
}

function onWindowResize(): void {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate(): void {
  requestAnimationFrame(animate);

  controls.update();
  renderer.render(scene, camera);

  // FPS counter
  frameCount++;
  const now = performance.now();
  if (now - lastTime >= 1000) {
    document.getElementById('stat-fps')!.textContent = frameCount.toString();
    frameCount = 0;
    lastTime = now;
  }
}

// Load PLY data into scene
function loadPLY(data: Uint8Array): void {
  const loader = new PLYLoader();

  // Convert Uint8Array to ArrayBuffer
  const buffer = data.buffer.slice(
    data.byteOffset,
    data.byteOffset + data.byteLength
  );

  const geometry = loader.parse(buffer);

  // Create point material
  const pointSize = parseFloat(
    (document.getElementById('point-size') as HTMLInputElement).value
  );

  const material = new THREE.PointsMaterial({
    size: pointSize * 0.01,
    vertexColors: true,
    sizeAttenuation: true,
  });

  // Remove existing point cloud
  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
    (pointCloud.material as THREE.Material).dispose();
  }

  // Create new point cloud
  pointCloud = new THREE.Points(geometry, material);

  // Center the point cloud
  geometry.computeBoundingBox();
  const center = new THREE.Vector3();
  geometry.boundingBox!.getCenter(center);
  geometry.translate(-center.x, -center.y, -center.z);

  scene.add(pointCloud);

  // Adjust camera
  const box = new THREE.Box3().setFromObject(pointCloud);
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  camera.position.set(0, 0, maxDim * 2);
  controls.target.set(0, 0, 0);
  controls.update();
}

// Update point size
function updatePointSize(size: number): void {
  if (pointCloud) {
    (pointCloud.material as THREE.PointsMaterial).size = size * 0.01;
  }
  document.getElementById('point-size-value')!.textContent = size.toString();
}

// Process image through API
async function processImage(
  imageData: ArrayBuffer,
  model: string,
  voxelSize: number
): Promise<PointCloudResult> {
  const apiUrl = (window as any).IRIS3D_API_URL || 'http://localhost:8080';

  const response = await fetch(
    `${apiUrl}/iris3d.v1.Iris3DService/ProcessImage`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_data: arrayBufferToBase64(imageData),
        format: 'jpeg',
        model: model,
        options: {
          downsampling: {
            enabled: true,
            voxel_size: voxelSize,
          },
          include_colors: true,
        },
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const result = await response.json();

  return {
    plyData: base64ToUint8Array(result.result.data),
    numPoints: result.result.num_points,
    stats: result.result.stats,
  };
}

// Utility functions
function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function downloadPLY(): void {
  if (!currentPlyData) return;

  const blob = new Blob([currentPlyData], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'iris3d_pointcloud.ply';
  a.click();
  URL.revokeObjectURL(url);
}

// Initialize UI
function initUI(): void {
  const imageInput = document.getElementById('image-input') as HTMLInputElement;
  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const pointSizeInput = document.getElementById('point-size') as HTMLInputElement;
  const voxelSizeInput = document.getElementById('voxel-size') as HTMLInputElement;
  const processBtn = document.getElementById('process-btn') as HTMLButtonElement;
  const downloadBtn = document.getElementById('download-btn') as HTMLButtonElement;
  const loadingDiv = document.getElementById('loading')!;

  let selectedImage: ArrayBuffer | null = null;

  // Image selection
  imageInput.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      selectedImage = await file.arrayBuffer();
      processBtn.disabled = false;
    }
  });

  // Point size slider
  pointSizeInput.addEventListener('input', (e) => {
    const size = parseFloat((e.target as HTMLInputElement).value);
    updatePointSize(size);
  });

  // Voxel size slider
  voxelSizeInput.addEventListener('input', (e) => {
    const size = parseFloat((e.target as HTMLInputElement).value);
    document.getElementById('voxel-size-value')!.textContent = size.toFixed(3);
  });

  // Process button
  processBtn.addEventListener('click', async () => {
    if (!selectedImage) return;

    processBtn.disabled = true;
    loadingDiv.classList.add('visible');

    try {
      const model = modelSelect.value;
      const voxelSize = parseFloat(voxelSizeInput.value);

      const result = await processImage(selectedImage, model, voxelSize);

      // Update stats
      document.getElementById('stat-points')!.textContent =
        result.numPoints.toLocaleString();
      document.getElementById('stat-inference')!.textContent =
        `${result.stats.inferenceTimeMs.toFixed(1)}ms`;
      document.getElementById('stat-total')!.textContent =
        `${result.stats.totalTimeMs.toFixed(1)}ms`;

      // Load point cloud
      currentPlyData = result.plyData;
      loadPLY(result.plyData);
      downloadBtn.disabled = false;
    } catch (error) {
      console.error('Processing error:', error);
      alert(`Error: ${error}`);
    } finally {
      processBtn.disabled = false;
      loadingDiv.classList.remove('visible');
    }
  });

  // Download button
  downloadBtn.addEventListener('click', downloadPLY);
}

// Main entry point
document.addEventListener('DOMContentLoaded', () => {
  initScene();
  initUI();
});
