const metricButtons = document.querySelectorAll('[data-metric]');
const deviceButtons = document.querySelectorAll('button[data-device]');
function fitTimelineLabels(plot) {
  const points = [...plot.querySelectorAll('svg a')];
  if (!points.length) return;
  const width = plot.clientWidth;
  const last = points[points.length - 1];
  // Keep the shared commit positions. Labels can move a few pixels at an edge.
  const bounds = points.map((point, index) => {
    const previousGap = index ? Number(point.dataset.position) - Number(points[index - 1].dataset.position) : 100;
    const nextGap = index < points.length - 1 ? Number(points[index + 1].dataset.position) - Number(point.dataset.position) : 100;
    point.querySelector('circle').style.setProperty('--point-radius', `${Math.min(previousGap, nextGap) * 0.35}cqi`);
    const position = Number(point.dataset.position) * width / 100;
    const texts = [...point.querySelectorAll('text')];
    const labelWidth = Math.max(...texts.map(text => text.getBBox().width));
    const anchor = index === 0 ? 'start' : point === last ? 'end' : 'middle';
    const offset = anchor === 'start' ? 0 : anchor === 'end' ? labelWidth : labelWidth / 2;
    const start = Math.max(2, Math.min(position - offset, width - labelWidth - 2));
    texts.forEach(text => {
      // CSS clamps the label within the current width between resize callbacks.
      text.setAttribute('x', '0');
      text.setAttribute('text-anchor', anchor);
      text.style.setProperty('--label-position', `${point.dataset.position}cqi`);
      text.style.setProperty('--label-width', `${labelWidth}px`);
      text.style.setProperty('--label-offset', `${offset}px`);
    });
    return {start, end: start + labelWidth, fits: labelWidth + 4 <= width};
  });
  const lastBounds = bounds[bounds.length - 1];
  let previousEnd = -Infinity;
  points.forEach((point, index) => {
    const {start, end, fits} = bounds[index];
    // Reserve the latest label first, including when a sparse series ends early.
    const show = fits && (point === last ||
      (start - previousEnd >= 12 && lastBounds.start - end >= 12));
    point.dataset.label = String(show);
    if (show) previousEnd = end;
  });
}
const timelineObserver = new ResizeObserver(entries => {
  // A font change can resize text while the plot stays the same size.
  const plots = new Set(entries.map(entry => entry.target.closest('.timeline-plot')));
  plots.forEach(fitTimelineLabels);
});
document.querySelectorAll('.timeline-plot').forEach(plot => {
  fitTimelineLabels(plot);
  timelineObserver.observe(plot);
  plot.querySelectorAll('svg text.timeline-label').forEach(text => timelineObserver.observe(text));
});
function setupTimelineReading(card) {
  const plot = card.querySelector('.timeline-plot');
  const points = [...plot.querySelectorAll('svg a')];
  if (!points.length) return;
  const latest = points[points.length - 1];
  const value = card.querySelector('[data-point-value]');
  const commit = card.querySelector('[data-point-commit]');
  const baseline = card.querySelector('[data-comparison="baseline"]');
  const previous = card.querySelector('[data-comparison="previous"]');
  const show = point => {
    points.forEach(item => { item.dataset.active = String(item === point); });
    value.textContent = `${point.dataset.value} bytes/s`;
    commit.textContent = `${point === latest ? 'Latest' : 'Commit'} · ${point.dataset.commit}`;
    baseline.querySelector('strong').textContent = point.dataset.baselineChange;
    baseline.title = `Baseline ${points[0].dataset.commit}: ${points[0].dataset.value} bytes/s`;
    const prior = points[points.indexOf(point) - 1];
    previous.querySelector('strong').textContent = point.dataset.previousChange || '—';
    previous.querySelector('small').textContent = prior ? 'vs previous' : 'first measurement';
    previous.title = prior ? `Previous ${prior.dataset.commit}: ${prior.dataset.value} bytes/s` : 'First measured commit';
  };
  const selectPoint = event => {
    const linkedPoint = event.target.closest('svg a');
    if (points.includes(linkedPoint)) {
      show(linkedPoint);
      return;
    }
    const bounds = plot.getBoundingClientRect();
    const position = (event.clientX - bounds.left) * 100 / bounds.width;
    const nearest = points.reduce((best, point) =>
      Math.abs(Number(point.dataset.position) - position) < Math.abs(Number(best.dataset.position) - position) ? point : best);
    show(nearest);
  };
  plot.addEventListener('pointermove', selectPoint);
  plot.addEventListener('pointerdown', selectPoint);
  card.addEventListener('pointerleave', event => {
    // Keep a tapped result readable after the finger lifts.
    if (event.pointerType !== 'touch') show(points.find(point => point === document.activeElement) || latest);
  });
  plot.addEventListener('focusin', event => { if (points.includes(event.target)) show(event.target); });
  plot.addEventListener('focusout', event => { if (!points.includes(event.relatedTarget)) show(latest); });
  show(latest);
}
document.querySelectorAll('.timeline-card').forEach(setupTimelineReading);
function fitResultTable(region) {
  if (!region.clientWidth) return;
  region.dataset.layout = 'table';
  region.dataset.layout = region.querySelector('table').scrollWidth > region.clientWidth + 1 ? 'cards' : 'table';
}
document.querySelectorAll('.results-details').forEach(details => {
  const region = details.querySelector('.table-scroll');
  let previousSize;
  let fitFrame;
  const refresh = () => { if (details.open) fitResultTable(region); };
  details.addEventListener('toggle', refresh);
  new ResizeObserver(entries => {
    // Card height can change during fitting. Only width or text size needs a new fit.
    const size = `${entries[0].contentRect.width}/${getComputedStyle(region).fontSize}`;
    if (size === previousSize) return;
    previousSize = size;
    cancelAnimationFrame(fitFrame);
    fitFrame = requestAnimationFrame(refresh);
  }).observe(region);
});
metricButtons.forEach(button => button.addEventListener('click', () => {
  const memory = button.dataset.metric === 'memory';
  metricButtons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  document.getElementById('speed-chart').hidden = memory;
  document.getElementById('memory-chart').hidden = !memory;
  document.getElementById('chart-title').textContent = memory ? 'Peak active MLX memory (MiB)' : 'Byte tokens per second';
  document.getElementById('chart-direction').textContent = memory ? 'LOWER USES LESS MEMORY' : 'HIGHER IS FASTER';
}));
deviceButtons.forEach(button => button.addEventListener('click', () => {
  const device = button.dataset.device;
  deviceButtons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  document.querySelectorAll('.bar-row, tbody tr').forEach(row => {
    row.hidden = device !== 'all' && row.dataset.device !== device;
  });
  const count = document.querySelectorAll('#speed-chart .bar-row:not([hidden])').length;
  document.getElementById('visible-count').textContent = `${count} measured configuration${count === 1 ? '' : 's'}`;
  document.querySelectorAll('.results-details[open] .table-scroll').forEach(fitResultTable);
}));
document.getElementById('copy-command').addEventListener('click', async event => {
  const button = event.currentTarget;
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(document.getElementById('benchmark-command').textContent);
    button.textContent = 'Copied';
    status.textContent = 'Benchmark command copied.';
  } catch {
    status.textContent = 'Select the command below and copy it with your keyboard.';
    button.textContent = 'Select text';
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('benchmark-command'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }
});
