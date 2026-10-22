'use strict';
const viewer = document.querySelector('#image-viewer');
const viewerImage = document.querySelector('#viewer-image');
if (typeof viewer.showModal === 'function') {
  const zoom = document.querySelector('#figure-zoom');
  const zoomValue = document.querySelector('#zoom-value');
  const viewport = viewer.querySelector('.viewer-content');
  function updateZoom() {
    viewerImage.style.width = `${zoom.value}%`;
    zoomValue.textContent = `${zoom.value}%`;
  }
  function resetZoom() {
    zoom.value = '100';
    updateZoom();
    viewport.scrollTop = 0;
    viewport.scrollLeft = 0;
  }
  document.querySelector('.viewer-tools').hidden = false;
  zoom.addEventListener('input', updateZoom);
  document.querySelector('#reset-zoom').addEventListener('click', resetZoom);
  document.querySelectorAll('[data-zoom]').forEach(link => {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      viewerImage.src = link.href;
      viewerImage.alt = link.querySelector('img').alt;
      document.querySelector('#original-image').href = link.href;
      viewer.showModal();
      document.body.classList.add('viewer-open');
      resetZoom();
    });
  });
  document.querySelector('#close-viewer').addEventListener('click', () => viewer.close());
  viewer.addEventListener('click', event => { if (event.target === viewer) viewer.close(); });
  viewer.addEventListener('close', () => document.body.classList.remove('viewer-open'));
}
const copyButton = document.querySelector('#copy-citation');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  const code = document.querySelector('#bibtex');
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(code.textContent);
    status.textContent = 'BibTeX copied to clipboard.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(code);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
