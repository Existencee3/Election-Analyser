const CANDIDATES = [
];

const PARTY_COLORS = {
  'BJP':  { bg:'#fde8e7', color:'#b52222', dot:'#b52222' },
  'INC':  { bg:'#e0f5ec', color:'#0a6640', dot:'#0a8754' },
  'AAP':  { bg:'#e8f0fd', color:'#1a3faa', dot:'#3b6dd9' },
  'SP':   { bg:'#fff3e0', color:'#a05a00', dot:'#c87000' },
  'BSP':  { bg:'#f3e8fd', color:'#6d28d9', dot:'#8b5cf6' },
  'TMC':  { bg:'#e8f5fd', color:'#0a5a8a', dot:'#0d82c8' },
  'DMK':  { bg:'#fff0e8', color:'#8a3a00', dot:'#c05500' },
  'NCP':  { bg:'#e8fbff', color:'#005a7a', dot:'#007aaa' },
  'YSRCP':{ bg:'#f0ffe8', color:'#3a7a00', dot:'#5aaa00' },
  'TDP':  { bg:'#fff8e8', color:'#7a5a00', dot:'#aa8000' },
  'IND':  { bg:'#f0f0f0', color:'#4a4a4a', dot:'#7a7a7a' },
};

const AVATAR_PALETTES = [
  ['#e8eaf5','#1a2a5e'],['#e0f5ec','#1a4a2e'],['#fde8e7','#5a1a1a'],
  ['#fff3e0','#5a2a00'],['#f3e8fd','#3a1a5e'],['#e8f5fd','#0a2a5e'],
  ['#fff8e8','#4a3a00'],['#f0ffe8','#1a3a00'],
];

function getPartyStyle(p) { return PARTY_COLORS[p] || { bg:'#f0f0f0', color:'#4a4a4a', dot:'#7a7a7a' }; }
function getAvatarStyle(n) { return AVATAR_PALETTES[n.charCodeAt(0) % AVATAR_PALETTES.length]; }
function initials(n) { const p = n.trim().split(' '); return p.length === 1 ? p[0].slice(0,2).toUpperCase() : (p[0][0] + p[p.length-1][0]).toUpperCase(); }

function buildDashboard() {
  const total = CANDIDATES.length;
  const winners = CANDIDATES.filter(c => c.result === 'won').length;
  const parties = new Set(CANDIDATES.map(c => c.party)).size;
  const states  = new Set(CANDIDATES.map(c => c.state)).size;
  const avgWin  = total > 0
    ? (CANDIDATES.filter(c => c.result === 'won').reduce((a,c) => a + (c.win_pct||0), 0) / (winners||1)).toFixed(1)
    : '—';

  document.getElementById('hs-total').textContent    = total || '—';
  document.getElementById('hs-winners').textContent  = winners || '—';
  document.getElementById('hs-parties').textContent  = parties || '—';
  document.getElementById('hs-states').textContent   = states  || '—';

  document.getElementById('sum-total').textContent   = total;
  document.getElementById('sum-winners').textContent = winners;
  document.getElementById('sum-parties').textContent = parties;
  document.getElementById('sum-states').textContent  = states;
  document.getElementById('sum-avgwin').textContent  = avgWin + '%';

  buildPartyChart();
  buildStateChart();
}

function buildPartyChart() {
  const counts = {};
  CANDIDATES.filter(c => c.result === 'won').forEach(c => counts[c.party] = (counts[c.party]||0)+1);
  const top = Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,8);
  if (!top.length) { document.getElementById('party-chart').innerHTML = '<div style="color:#9a9a9a;font-size:.8rem;">No data yet.</div>'; return; }
  const max = top[0][1];
  const colors = ['#1a2a5e','#2a4a8e','#3a6abe','#4a8aee','#1a4a2e','#2a6a50','#8a3a00','#6d28d9'];
  document.getElementById('party-chart').innerHTML = top.map(([p,n],i) =>
    `<div class="bar-row">
      <div class="bar-label" title="${p}">${p.length>13?p.slice(0,13)+'…':p}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(n/max*100).toFixed(1)}%;background:${colors[i%colors.length]}"></div></div>
      <div class="bar-val">${n}</div>
    </div>`
  ).join('');
}

function buildStateChart() {
  const counts = {};
  CANDIDATES.filter(c => c.result === 'won').forEach(c => counts[c.state] = (counts[c.state]||0)+1);
  const top = Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,8);
  if (!top.length) { document.getElementById('state-chart').innerHTML = '<div style="color:#9a9a9a;font-size:.8rem;">No data yet.</div>'; return; }
  const max = top[0][1];
  const colors = ['#1a4a2e','#2a6a50','#1a2a5e','#2a4a8e','#8a3a00','#c87000','#6d28d9','#3a7a00'];
  document.getElementById('state-chart').innerHTML = top.map(([s,n],i) =>
    `<div class="bar-row">
      <div class="bar-label" title="${s}">${s.length>14?s.slice(0,14)+'…':s}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(n/max*100).toFixed(1)}%;background:${colors[i%colors.length]}"></div></div>
      <div class="bar-val">${n}</div>
    </div>`
  ).join('');
}

let filtered = [...CANDIDATES], currentPage = 1;
const PER_PAGE = 24;

function buildFilters() {
  const parties = [...new Set(CANDIDATES.map(c => c.party))].sort();
  const pf = document.getElementById('partyFilter');
  parties.forEach(p => { const o = document.createElement('option'); o.value = p; o.textContent = p; pf.appendChild(o); });

  const states = [...new Set(CANDIDATES.map(c => c.state))].sort();
  const sf = document.getElementById('stateFilter');
  states.forEach(s => { const o = document.createElement('option'); o.value = s; o.textContent = s; sf.appendChild(o); });
}

function filterCandidates() {
  const q      = document.getElementById('searchInput').value.toLowerCase();
  const party  = document.getElementById('partyFilter').value;
  const state  = document.getElementById('stateFilter').value;
  const result = document.getElementById('resultFilter').value;

  filtered = CANDIDATES.filter(c => {
    if (q && !c.name.toLowerCase().includes(q) && !c.party.toLowerCase().includes(q) && !c.constituency.toLowerCase().includes(q)) return false;
    if (party  && c.party  !== party)  return false;
    if (state  && c.state  !== state)  return false;
    if (result && c.result !== result) return false;
    return true;
  });
  currentPage = 1;
  renderCards();
}

function resultClass(r) {
  if (r === 'won')    return 'winner';
  if (r === 'runner') return 'runner';
  return 'lost';
}

function resultLabel(r) {
  if (r === 'won')    return '✓ Won';
  if (r === 'runner') return '↑ Runner-up';
  return '✗ Lost';
}

function renderCards() {
  const grid = document.getElementById('candidates-grid');
  const countEl = document.getElementById('results-count');
  countEl.textContent = `${filtered.length} candidates`;

  if (CANDIDATES.length === 0) {
    grid.innerHTML = `
      <div class="empty-state" style="grid-column:1/-1;">
        <div class="empty-state-icon">🗳️</div>
        <div class="empty-state-title">Data Not Loaded Yet</div>
        <div class="empty-state-desc">Add your All-India election dataset to the <code>CANDIDATES</code> array in <strong>js/allindia.js</strong>. Each record needs: name, party, constituency, state, votes, total_votes, win_pct, result, year.</div>
        <div class="empty-state-code">
          { id: 1,<br>
          &nbsp;&nbsp;name: "Candidate Name",<br>
          &nbsp;&nbsp;party: "BJP",<br>
          &nbsp;&nbsp;constituency: "Constituency",<br>
          &nbsp;&nbsp;state: "State",<br>
          &nbsp;&nbsp;votes: 500000,<br>
          &nbsp;&nbsp;total_votes: 1000000,<br>
          &nbsp;&nbsp;win_pct: 50.0,<br>
          &nbsp;&nbsp;result: "won",<br>
          &nbsp;&nbsp;year: 2024 }
        </div>
      </div>`;
    document.getElementById('pagination').innerHTML = '';
    return;
  }

  const start = (currentPage - 1) * PER_PAGE;
  const slice = filtered.slice(start, start + PER_PAGE);

  if (slice.length === 0) {
    grid.innerHTML = `<div style="grid-column:1/-1;text-align:center;padding:3rem;color:var(--ink-faint)">No candidates found.</div>`;
    document.getElementById('pagination').innerHTML = '';
    return;
  }

  const rc = resultClass;
  const rl = resultLabel;

  grid.innerHTML = slice.map(c => {
    const [ab, ac] = getAvatarStyle(c.name);
    const ps = getPartyStyle(c.party);
    const rClass = rc(c.result);
    const rLabel = rl(c.result);
    const pct = typeof c.win_pct === 'number' ? c.win_pct.toFixed(1) : '—';
    return `<div class="win-card">
      <div class="win-card-top">
        <div class="win-card-name">${c.name}</div>
        <div class="win-pct-badge ${rClass}">${pct}%</div>
      </div>
      <div class="win-card-meta">
        <span class="meta-pill meta-party" style="background:${ps.bg};color:${ps.color}">${c.party}</span>
        <span class="meta-pill meta-constituency">${c.constituency||'—'}</span>
        <span class="meta-pill meta-state">${c.state||'—'}</span>
      </div>
      <div class="win-bar-wrap">
        <div class="win-bar-label">
          <span>${rLabel}</span>
          <span style="font-family:'DM Mono',monospace">${pct}% of votes</span>
        </div>
        <div class="win-bar-track">
          <div class="win-bar-fill ${rClass}" style="width:${Math.min(c.win_pct||0, 100)}%"></div>
        </div>
      </div>
    </div>`;
  }).join('');

  renderPagination();
}

function renderPagination() {
  const total = Math.ceil(filtered.length / PER_PAGE);
  const pag = document.getElementById('pagination');
  if (total <= 1) { pag.innerHTML = ''; return; }
  let btns = `<button class="page-btn" onclick="goPage(${currentPage-1})" ${currentPage===1?'disabled':''}>‹</button>`;
  for (let i = 1; i <= total; i++) {
    if (i===1||i===total||Math.abs(i-currentPage)<=2)
      btns += `<button class="page-btn ${i===currentPage?'active':''}" onclick="goPage(${i})">${i}</button>`;
    else if (Math.abs(i-currentPage)===3)
      btns += `<span style="padding:.5rem .3rem;color:var(--ink-faint)">…</span>`;
  }
  btns += `<button class="page-btn" onclick="goPage(${currentPage+1})" ${currentPage===total?'disabled':''}>›</button>`;
  pag.innerHTML = btns;
}

function goPage(p) {
  const total = Math.ceil(filtered.length / PER_PAGE);
  if (p < 1 || p > total) return;
  currentPage = p;
  renderCards();
  document.getElementById('candidates-grid').scrollIntoView({ behavior:'smooth', block:'start' });
}

buildDashboard();
if (CANDIDATES.length > 0) {
  buildFilters();
  filterCandidates();
} else {
  renderCards();
}
