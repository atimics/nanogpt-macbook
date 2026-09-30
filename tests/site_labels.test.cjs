const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');

const script = fs.readFileSync(path.join(__dirname, '../site/app.js'), 'utf8');

function render(positions, width, textWidths = [32, 43]) {
  const points = positions.map((position, index) => ({
    dataset: {position: String(position), label: String(index === 0 || index === positions.length - 1)},
    texts: textWidths.map(labelWidth => ({
      attrs: {
        x: String(position * width / 100),
        'text-anchor': index === 0 ? 'start' : index === positions.length - 1 ? 'end' : 'middle',
      },
      getBBox: () => ({width: labelWidth}),
      setAttribute(name, value) { this.attrs[name] = value; },
    })),
    querySelectorAll() { return this.texts; },
  }));
  const plot = {clientWidth: width, querySelectorAll: () => points};
  let resize;
  vm.runInNewContext(script, {
    document: {
      querySelectorAll: selector => selector === '.timeline-plot' ? [plot] : [],
      getElementById: () => ({addEventListener() {}}),
    },
    ResizeObserver: class {
      constructor(callback) { resize = callback; }
      observe() {}
    },
  });
  return {points, resize(width) { plot.clientWidth = width; resize([{target: plot}]); }};
}

function checkBounds(points, width) {
  let previousRight = -Infinity;
  for (const point of points.filter(point => point.dataset.label === 'true')) {
    const boxes = point.texts.map(text => {
      const x = Number(text.attrs.x), size = text.getBBox().width;
      const anchor = text.attrs['text-anchor'];
      const left = x - (anchor === 'end' ? size : anchor === 'middle' ? size / 2 : 0);
      return {left, right: left + size};
    });
    const left = Math.min(...boxes.map(box => box.left));
    const right = Math.max(...boxes.map(box => box.right));
    assert.ok(left >= 0 && right <= width, `${left}..${right} must fit ${width}px`);
    assert.ok(left - previousRight >= 12, 'Visible label groups need a clear gap');
    previousRight = right;
  }
  assert.equal(points.at(-1).dataset.label, 'true', 'Keep the latest label visible');
}

test('sparse history fits when later commits measure another device', () => {
  const positions = Array.from({length: 8}, (_, i) => 9 + i * 82 / 70);
  const {points} = render(positions, 244);
  checkBounds(points, 244);
  assert.deepEqual(points.map(point => Number(point.dataset.position)), positions);
});

test('a single measurement fits at either end of the shared axis', () => {
  for (const position of [9, 91]) checkBounds(render([position], 244).points, 244);
});

test('dense history fits through repeated width changes', () => {
  const positions = Array.from({length: 60}, (_, i) => 9 + i * 82 / 59);
  const chart = render(positions, 244);
  for (const width of [244, 502, 204, 670, 244]) {
    chart.resize(width);
    checkBounds(chart.points, width);
    assert.equal(chart.points.length, 60);
  }
});

test('larger rendered glyphs keep the latest label clear', () => {
  const {points} = render([9, 12, 20, 25], 244, [64, 86]);
  checkBounds(points, 244);
});

test('result layout measures the table after resize, reopen, and filtering', () => {
  const context = {
    document: {
      querySelectorAll: () => [],
      getElementById: () => ({addEventListener() {}}),
    },
    ResizeObserver: class {},
  };
  vm.runInNewContext(script, context);
  let needed = 1190;
  const region = {
    clientWidth: 1104,
    dataset: {layout: 'cards'},
    querySelector: () => ({get scrollWidth() {
      return region.dataset.layout === 'cards' ? region.clientWidth : needed;
    }}),
  };
  for (const width of [1104, 320, 1280, 0, 1280, 320, 1104]) {
    region.clientWidth = width;
    context.fitResultTable(region);
    if (width) assert.equal(region.dataset.layout, width < needed ? 'cards' : 'table');
  }
  needed = 1000; // A device filter can reduce the table's width.
  context.fitResultTable(region);
  assert.equal(region.dataset.layout, 'table');
});

test('result resizing fits outside the observer cycle and ignores height changes', () => {
  let observer;
  let frame;
  let measures = 0;
  const region = {
    clientWidth: 320,
    dataset: {layout: 'cards'},
    querySelector: () => ({get scrollWidth() { measures++; return 1000; }}),
  };
  const details = {open: true, querySelector: () => region, addEventListener() {}};
  const context = {
    document: {
      querySelectorAll: selector => selector === '.results-details' ? [details] : [],
      getElementById: () => ({addEventListener() {}}),
    },
    getComputedStyle: () => ({fontSize: '20px'}),
    ResizeObserver: class { constructor(callback) { observer = callback; } observe() {} },
    requestAnimationFrame: callback => { frame = callback; return 1; },
    cancelAnimationFrame() { frame = undefined; },
  };
  vm.runInNewContext(script, context);
  observer([{contentRect: {width: 320, height: 500}}]);
  assert.equal(measures, 0);
  frame();
  assert.equal(measures, 1);
  assert.equal(region.dataset.layout, 'cards');
  frame = undefined;
  observer([{contentRect: {width: 320, height: 1000}}]);
  assert.equal(frame, undefined);
  region.clientWidth = 1280;
  observer([{contentRect: {width: 1280, height: 1000}}]);
  frame();
  assert.equal(region.dataset.layout, 'table');
});
