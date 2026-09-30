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
      width: labelWidth,
      vars: {},
      attrs: {
        x: String(position * width / 100),
        'text-anchor': index === 0 ? 'start' : index === positions.length - 1 ? 'end' : 'middle',
      },
      getBBox() { return {width: this.width}; },
      setAttribute(name, value) { this.attrs[name] = value; },
    })),
    querySelectorAll() { return this.texts; },
    querySelector() { return this.circle; },
  }));
  const plot = {
    clientWidth: width,
    querySelectorAll: selector => selector === 'svg a' ? points : points.flatMap(point => point.texts),
    closest() { return this; },
  };
  points.flatMap(point => point.texts).forEach(text => {
    text.style = {setProperty(name, value) { text.vars[name] = value; }};
    text.closest = () => plot;
  });
  points.forEach(point => {
    point.circle = {vars: {}};
    point.circle.style = {setProperty(name, value) { point.circle.vars[name] = value; }};
  });
  let resize;
  const observed = new Set();
  vm.runInNewContext(script, {
    document: {
      querySelectorAll: selector => selector === '.timeline-plot' ? [plot] : [],
      getElementById: () => ({addEventListener() {}}),
    },
    ResizeObserver: class {
      constructor(callback) { resize = callback; }
      observe(target) { observed.add(target); }
    },
  });
  return {
    points,
    resize(width) { plot.clientWidth = width; resize([{target: plot}]); },
    textResize() {
      const text = points[0].texts[0];
      assert.ok(observed.has(text), 'Observe label dimensions as well as plot dimensions');
      resize([{target: text}]);
    },
  };
}

function checkBounds(points, width) {
  let previousRight = -Infinity;
  for (const point of points.filter(point => point.dataset.label === 'true')) {
    const boxes = point.texts.map(text => {
      const reserved = parseFloat(text.vars['--label-width']);
      const offset = parseFloat(text.vars['--label-offset']);
      const position = parseFloat(text.vars['--label-position']) * width / 100;
      const x = Math.max(2, Math.min(position - offset, width - reserved - 2)) + offset;
      const size = text.getBBox().width;
      assert.equal(text.attrs.x, '0');
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

test('font changes refit labels while the plot width stays the same', () => {
  const chart = render([9, 25, 41, 57, 73, 91], 244);
  const originalVisible = chart.points.filter(point => point.dataset.label === 'true').length;
  chart.points.flatMap(point => point.texts).forEach(text => { text.width *= 2.4; });
  chart.textResize();
  checkBounds(chart.points, 244);
  assert.ok(chart.points.filter(point => point.dataset.label === 'true').length < originalVisible);
  assert.equal(chart.points.length, 6);
});

test('dense markers keep space between neighboring commits through resize', () => {
  const positions = Array.from({length: 82}, (_, i) => 9 + i * 82 / 81);
  const chart = render(positions, 260);
  for (const width of [502, 260, 186, 502]) {
    chart.resize(width);
    const radii = chart.points.map(point => Math.min(5, parseFloat(point.circle.vars['--point-radius']) * width / 100));
    for (let i = 1; i < positions.length; i++) {
      assert.ok(radii[i - 1] + radii[i] < (positions[i] - positions[i - 1]) * width / 100);
    }
    assert.deepEqual(chart.points.map(point => Number(point.dataset.position)), positions);
  }
});

test('point reading follows pointer and focus, then returns to latest', () => {
  const document = {querySelectorAll: () => [], getElementById: () => ({addEventListener() {}})};
  const context = {document, ResizeObserver: class {}};
  vm.runInNewContext(script, context);
  const points = [9, 50, 91].map((position, index) => ({
    dataset: {
      position: String(position), value: String((index + 1) * 100), commit: `commit${index}`,
      baselineChange: ['+0.0%', '+100.0%', '+200.0%'][index],
      previousChange: ['', '+100.0%', '+50.0%'][index],
    },
  }));
  const events = {};
  const plot = {
    querySelectorAll: () => points,
    getBoundingClientRect: () => ({left: 10, width: 200}),
    addEventListener(name, handler) { events[name] = handler; },
  };
  const value = {}, commit = {};
  const comparison = () => ({strong: {}, small: {}, querySelector(name) { return this[name]; }});
  const baseline = comparison(), previous = comparison();
  const nodes = {
    '.timeline-plot': plot, '[data-point-value]': value, '[data-point-commit]': commit,
    '[data-comparison="baseline"]': baseline, '[data-comparison="previous"]': previous,
  };
  context.setupTimelineReading({
    querySelector: selector => nodes[selector],
    addEventListener(name, handler) { events[`card:${name}`] = handler; },
  });
  assert.equal(value.textContent, '300 bytes/s');
  assert.equal(commit.textContent, 'Latest · commit2');
  assert.equal(baseline.strong.textContent, '+200.0%');
  assert.equal(previous.strong.textContent, '+50.0%');
  assert.equal(baseline.title, 'Baseline commit0: 100 bytes/s');
  assert.equal(previous.title, 'Previous commit1: 200 bytes/s');
  events.pointerdown({clientX: 110, target: {closest: () => null}});
  assert.equal(value.textContent, '200 bytes/s');
  assert.equal(commit.textContent, 'Commit · commit1');
  assert.equal(baseline.strong.textContent, '+100.0%');
  assert.equal(previous.strong.textContent, '+100.0%');
  assert.equal(previous.title, 'Previous commit0: 100 bytes/s');
  assert.equal(points.filter(point => point.dataset.active === 'true').length, 1);
  events.pointermove({clientX: 110, target: {closest: () => points[0]}});
  assert.equal(value.textContent, '100 bytes/s', 'A hovered hash belongs to its linked point');
  assert.equal(baseline.strong.textContent, '+0.0%');
  assert.equal(previous.strong.textContent, '—');
  assert.equal(previous.small.textContent, 'first measurement');
  document.activeElement = points[0];
  events.focusin({target: points[0]});
  events['card:pointerleave']({pointerType: 'mouse'});
  assert.equal(value.textContent, '100 bytes/s');
  document.activeElement = null;
  events.focusout({relatedTarget: null});
  assert.equal(value.textContent, '300 bytes/s');
  events.pointermove({clientX: 110, target: {closest: () => null}});
  events['card:pointerleave']({pointerType: 'touch'});
  assert.equal(commit.textContent, 'Commit · commit1', 'Keep a touch selection after finger lift');
  assert.equal(events.pointerleave, undefined, 'Keep the selection while reading its comparison titles');
  events['card:pointerleave']({pointerType: 'mouse'});
  assert.equal(commit.textContent, 'Latest · commit2');
  assert.equal(baseline.strong.textContent, '+200.0%');
  assert.equal(previous.strong.textContent, '+50.0%');
  assert.equal(previous.small.textContent, 'vs previous');
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
