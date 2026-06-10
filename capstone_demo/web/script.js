const STATUS_URL = '/api/status';
const REFRESH_INTERVAL = 1000;

const selectedCameraEl = document.getElementById('selected-camera');
const streamStatusEl = document.getElementById('stream-status');
const yoloChipEl = document.getElementById('yolo-chip');
const roiPathEl = document.getElementById('roi-path');
const trafficCountEl = document.getElementById('traffic-count');
const trafficDownEl = document.getElementById('traffic-down');
const trafficUpEl = document.getElementById('traffic-up');
const downboundTrafficEl = document.getElementById('downbound-traffic');
const downboundAccidentEl = document.getElementById('downbound-accident');
const upboundTrafficEl = document.getElementById('upbound-traffic');
const upboundAccidentEl = document.getElementById('upbound-accident');
const downChartEl = document.getElementById('traffic-down-chart');
const upChartEl = document.getElementById('traffic-up-chart');

async function fetchJson(url) {
  try {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (error) {
    console.error('Fetch error', url, error);
    return null;
  }
}

function updateStatus(status) {
  if (!status) return;

  selectedCameraEl.textContent = status.selected_name || 'USB Camera';
  streamStatusEl.textContent = status.stream_status || '대기 중';
  roiPathEl.textContent = status.roi_path || 'ROI 파일 없음';
  trafficCountEl.textContent = String(status.traffic_count ?? 0);
  trafficDownEl.textContent = String(status.traffic_down ?? 0);
  trafficUpEl.textContent = String(status.traffic_up ?? 0);
  downboundTrafficEl.textContent = status.congestion_down ? '복잡' : '원활';
  upboundTrafficEl.textContent = status.congestion_up ? '복잡' : '원활';
  downboundAccidentEl.textContent = status.congestion_down ? '발생' : '미발생';
  upboundAccidentEl.textContent = status.congestion_up ? '발생' : '미발생';

  if (status.yolo_enabled) {
    yoloChipEl.classList.remove('is-off');
    yoloChipEl.classList.add('is-on');
    yoloChipEl.textContent = 'YOLO ON';
  } else {
    yoloChipEl.classList.remove('is-on');
    yoloChipEl.classList.add('is-off');
    yoloChipEl.textContent = 'YOLO OFF';
  }

  renderCharts({
    trafficUpHistory: status.traffic_up_history || [0],
    trafficDownHistory: status.traffic_down_history || [0],
    trafficCount: status.traffic_count || 0,
  });
}

function fitCanvas(canvas) {
  if (!canvas) return null;
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

function drawLineChart(canvas, values, options) {
  const ctx = fitCanvas(canvas);
  if (!ctx) return;

  const width = canvas.getBoundingClientRect().width;
  const height = canvas.getBoundingClientRect().height;
  const padding = { top: 18, right: 12, bottom: 24, left: 12 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const maxValue = Math.max(6, ...values, options.maxFloor || 0);
  const points = values.length > 1 ? values : [0, values[0] || 0];

  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = options.gridColor;
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i += 1) {
    const y = padding.top + (innerHeight / 3) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
  gradient.addColorStop(0, options.fillTop);
  gradient.addColorStop(1, options.fillBottom);

  ctx.beginPath();
  points.forEach((value, index) => {
    const x = padding.left + (innerWidth * index) / Math.max(1, points.length - 1);
    const y = padding.top + innerHeight - (Math.min(value, maxValue) / maxValue) * innerHeight;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.closePath();

  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.beginPath();
  points.forEach((value, index) => {
    const x = padding.left + (innerWidth * index) / Math.max(1, points.length - 1);
    const y = padding.top + innerHeight - (Math.min(value, maxValue) / maxValue) * innerHeight;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = options.lineColor;
  ctx.lineWidth = 3;
  ctx.stroke();
}

function renderCharts({ trafficCount, trafficUpHistory, trafficDownHistory }) {
  drawLineChart(downChartEl, trafficDownHistory, {
    lineColor: '#ffae42',
    fillTop: 'rgba(255, 174, 66, 0.32)',
    fillBottom: 'rgba(255, 174, 66, 0.02)',
    gridColor: 'rgba(160, 184, 220, 0.14)',
    maxFloor: trafficCount + 2,
  });

  drawLineChart(upChartEl, trafficUpHistory, {
    lineColor: '#3cf28a',
    fillTop: 'rgba(60, 242, 138, 0.32)',
    fillBottom: 'rgba(60, 242, 138, 0.02)',
    gridColor: 'rgba(160, 184, 220, 0.14)',
    maxFloor: trafficCount + 2,
  });
}

async function refresh() {
  const status = await fetchJson(STATUS_URL);
  if (status) updateStatus(status);
}

window.addEventListener('load', () => {
  refresh();
  setInterval(refresh, REFRESH_INTERVAL);
});
