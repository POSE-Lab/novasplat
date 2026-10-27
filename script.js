'use strict';
const slider = document.querySelector('#refinement-slider');
const comparison = document.querySelector('#refinement-image');
const playback = document.querySelector('#toggle-refinement');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let playing = !reducedMotion.matches;
let frame = null;
let previousTime = null;
let progress = 0.5;
let direction = 1;
function updateComparison() {
  comparison.style.setProperty('--reveal', `${slider.value}%`);
  slider.setAttribute('aria-valuetext', `${Math.round(slider.value)}% refined predictions revealed`);
}
function animate(time) {
  if (!playing) return;
  if (previousTime !== null) {
    progress += direction * Math.min(time - previousTime, 80) / 4000;
    if (progress >= 1) { progress = 1; direction = -1; }
    if (progress <= 0) { progress = 0; direction = 1; }
  }
  previousTime = time;
  // Smoothly ease at each end of the sweep.
  slider.value = String((0.5 - 0.5 * Math.cos(Math.PI * progress)) * 100);
  updateComparison();
  frame = requestAnimationFrame(animate);
}
function setPlayback(next) {
  playing = next;
  playback.textContent = playing ? 'Pause animation' : 'Play animation';
  previousTime = null;
  cancelAnimationFrame(frame);
  if (playing) frame = requestAnimationFrame(animate);
}
slider.step = 'any';
slider.addEventListener('input', () => {
  setPlayback(false);
  progress = Math.acos(1 - 2 * Number(slider.value) / 100) / Math.PI;
  updateComparison();
});
playback.hidden = false;
playback.addEventListener('click', () => setPlayback(!playing));
reducedMotion.addEventListener('change', event => { if (event.matches) setPlayback(false); });
document.addEventListener('visibilitychange', () => { previousTime = null; });
updateComparison();
setPlayback(playing);

const methods = [
  ['original', 'Original'], ['nova_full', 'NovaSplat (Ours)'],
  ['gfreedet', 'GFreeDet2'], ['opformer', 'OPFormer'], ['nemo', 'NeMO'],
];
const scenes = window.NOVASPLAT_SCENES || [];
/* Previous selector implementation, preserved to compare layouts.
const sceneSelect = document.querySelector('#scene-select');
const gallery = document.querySelector('#gallery');
function renderScene() {
  const scene = scenes[Number(sceneSelect.value)];
  gallery.replaceChildren();
  for (const [method, label] of methods) {
    const figure = document.createElement('figure');
    const img = document.createElement('img');
    img.src = `${scene.prefix}${method}.webp`;
    img.alt = `${label}, ${scene.label}`;
    img.width = 480;
    img.height = 270;
    img.loading = 'lazy';
    const caption = document.createElement('figcaption');
    caption.textContent = label;
    if (method === 'nova_full') figure.className = 'our-result';
    figure.append(img, caption);
    gallery.append(figure);
  }
  document.querySelector('#gallery-status').textContent = `Showing ${scene.label}. Bounding cuboids highlight the predicted 6D poses.`;
}
if (scenes.length) {
  document.querySelector('#additional-scenes').hidden = false;
  scenes.forEach((scene, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = scene.label;
    sceneSelect.append(option);
  });
  sceneSelect.addEventListener('change', renderScene);
  renderScene();
}

*/

// Active layout: all scene comparisons in a vertically scrollable region.
if (scenes.length) {
  const headings = document.querySelector('#scenes-column-headings');
  methods.forEach(([, label]) => {
    const heading = document.createElement('span');
    heading.textContent = label;
    headings.append(heading);
  });
  const list = document.querySelector('#scenes-list');
  scenes.forEach(scene => {
    const section = document.createElement('section');
    section.className = 'scene-comparison-row';
    section.setAttribute('aria-label', scene.label);
    const row = document.createElement('div');
    row.className = 'scene-comparison-images';
    methods.forEach(([method, label]) => {
      const image = document.createElement('img');
      image.src = `${scene.prefix}${method}.webp`;
      image.alt = `${label}, HOPEv2 ${scene.label}`;
      image.width = 480;
      image.height = 270;
      image.loading = 'lazy';
      row.append(image);
    });
    section.append(row);
    list.append(section);
  });
  document.querySelector('#additional-scenes').hidden = false;
}

const copyButton = document.querySelector('#copy-citation');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    status.textContent = 'BibTeX copied to clipboard.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'BibTeX selected. Press Ctrl+C (or Command+C) to copy.';
  }
});

// Introductory examples: manual scrolling, dragging, and dot navigation.
const exampleCarousel = document.querySelector('.example-carousel');
if (exampleCarousel) {
  const viewport = exampleCarousel.querySelector('.example-viewport');
  const pages = Array.from(exampleCarousel.querySelectorAll('.example-page'));
  const dots = Array.from(exampleCarousel.querySelectorAll('.example-dot'));
  function showPage(index) {
    const page = Math.max(0, Math.min(pages.length - 1, index));
    viewport.scrollTo({ left: page * viewport.clientWidth, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
  }
  function updateDots() {
    const page = Math.round(viewport.scrollLeft / Math.max(1, viewport.clientWidth));
    dots.forEach((dot, i) => {
      if (i === page) dot.setAttribute('aria-current', 'true');
      else dot.removeAttribute('aria-current');
    });
  }
  dots.forEach(dot => dot.addEventListener('click', () => showPage(Number(dot.dataset.page))));
  viewport.addEventListener('scroll', updateDots, { passive: true });
  exampleCarousel.addEventListener('keydown', event => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault();
      const page = Math.round(viewport.scrollLeft / Math.max(1, viewport.clientWidth));
      showPage(page + (event.key === 'ArrowRight' ? 1 : -1));
    }
  });
  let wheelTimer;
  // A standard mouse wheel moves the strip horizontally while there are more examples.
  viewport.addEventListener('wheel', event => {
    if (event.ctrlKey || event.shiftKey || Math.abs(event.deltaX) >= Math.abs(event.deltaY)) return;
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport.clientWidth : 1);
    const max = viewport.scrollWidth - viewport.clientWidth;
    if ((delta > 0 && viewport.scrollLeft < max - 1) || (delta < 0 && viewport.scrollLeft > 1)) {
      event.preventDefault();
      viewport.classList.add('is-wheeling');
      viewport.scrollLeft += delta;
      clearTimeout(wheelTimer);
      wheelTimer = setTimeout(() => viewport.classList.remove('is-wheeling'), 180);
    }
  }, { passive: false });
  let drag = null;
  viewport.addEventListener('pointerdown', event => {
    if (event.pointerType !== 'mouse' || event.button !== 0) return;
    drag = { x: event.clientX, left: viewport.scrollLeft };
    viewport.setPointerCapture(event.pointerId);
    viewport.classList.add('is-dragging');
  });
  viewport.addEventListener('pointermove', event => {
    if (drag) viewport.scrollLeft = drag.left - (event.clientX - drag.x);
  });
  function endDrag(event) {
    if (!drag) return;
    drag = null;
    viewport.classList.remove('is-dragging');
    if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
    showPage(Math.round(viewport.scrollLeft / Math.max(1, viewport.clientWidth)));
  }
  viewport.addEventListener('pointerup', endDrag);
  viewport.addEventListener('pointercancel', endDrag);
  viewport.addEventListener('dragstart', event => event.preventDefault());
  pages.forEach(page => page.removeAttribute('aria-hidden'));
  exampleCarousel.querySelector('.example-controls').hidden = false;
  updateDots();
}
