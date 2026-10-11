'use strict';
const cards = [...document.querySelectorAll('.sample')];
const language = document.getElementById('language');
const search = document.getElementById('search');
function filter() {
  if (!language || !search) return;
  const query = search.value.trim().toLowerCase();
  let shown = 0;
  for (const card of cards) {
    const show = (language.value === 'all' || card.dataset.language === language.value) && (!query || card.dataset.search.includes(query));
    card.classList.toggle('hidden', !show);
    if (show) shown++;
  }
  document.getElementById('visible-count').textContent = `${shown} of ${cards.length} recordings`;
}
language?.addEventListener('change', filter);
search?.addEventListener('input', filter);
document.addEventListener('click', event => {
  const seek = event.target.closest('[data-start]');
  if (!seek) return;
  const audio = seek.closest('.sample')?.querySelector('audio');
  if (!audio) return;
  const start = Number(seek.dataset.start);
  const play = () => { audio.currentTime = Math.max(0, start); audio.play().catch(() => {}); };
  if (audio.readyState === 0) { audio.addEventListener('loadedmetadata', play, {once: true}); audio.load(); }
  else play();
});
document.addEventListener('keydown', event => {
  if ((event.key === 'Enter' || event.key === ' ') && event.target.matches('svg [data-start]')) {
    event.preventDefault(); event.target.dispatchEvent(new MouseEvent('click', {bubbles: true}));
  }
});
document.addEventListener('play', event => {
  if (event.target.tagName !== 'AUDIO') return;
  document.querySelectorAll('audio').forEach(audio => { if (audio !== event.target) audio.pause(); });
}, true);
filter();
