const metricButtons = document.querySelectorAll('[data-metric]');
const deviceButtons = document.querySelectorAll('button[data-device]');
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
