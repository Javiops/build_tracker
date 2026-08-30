const $ = (id) => document.getElementById(id);

let ddragon = "16.17.1";
const urlPuuid = new URLSearchParams(location.search).get("puuid");
let selectedPuuid = urlPuuid || "";
let selectedName = "this player";

function champImg(name) {
  return `https://ddragon.leagueoflegends.com/cdn/${ddragon}/img/champion/${name}.png`;
}

function itemImg(id) {
  return `https://ddragon.leagueoflegends.com/cdn/${ddragon}/img/item/${id}.png`;
}

function iconImg(id) {
  return `https://ddragon.leagueoflegends.com/cdn/${ddragon}/img/profileicon/${id}.png`;
}

function fmtTime(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

function fmtDate(ms) {
  return new Date(ms).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function setLog(text) {
  $("log").textContent = text;
}

function renderChip(player, games) {
  const chip = $("player-chip");
  if (!chip) return;
  if (!player) {
    chip.innerHTML = `<div class="icon-fallback">?</div><div><strong>Pick a player</strong><span>Search the KR corpus</span></div>`;
    selectedName = "this player";
    return;
  }
  selectedName = player.game_name || player.label || "this player";
  const initial = (selectedName[0] || "?").toUpperCase();
  const icon = player.profile_icon_id
    ? `<img src="${iconImg(player.profile_icon_id)}" alt="" />`
    : `<div class="icon-fallback">${initial}</div>`;
  const riotId = `${player.game_name || ""}#${player.tag_line || ""}`.replace(/^#/, "");
  const extra = games != null ? ` · ${games} games` : "";
  chip.innerHTML = `${icon}<div><strong>${player.label || riotId}</strong><span>${riotId}${extra} · click to show all</span></div>`;
}

async function loadStatus() {
  const qs = selectedPuuid ? `?puuid=${encodeURIComponent(selectedPuuid)}` : "";
  const res = await fetch(`/api/status${qs}`);
  const data = await res.json();
  ddragon = data.ddragon_version || ddragon;
  $("stat-games").textContent = data.unique_games ?? data.match_count;
  if ($("stat-perspectives")) $("stat-perspectives").textContent = data.match_count ?? 0;
  $("stat-visits").textContent = data.visit_count ?? data.decision_count ?? 0;
  $("stat-sync").textContent = data.last_sync
    ? new Date(data.last_sync).toLocaleString()
    : "never";
  if (selectedPuuid && data.player) {
    renderChip(data.player, data.player_games);
  } else if (!selectedPuuid) {
    renderChip(null);
  }
  if (!data.has_api_key) {
    setLog("No API key loaded. Copy .env.example to .env, paste your Riot key, restart, then sync.");
    $("sync-btn").disabled = true;
  } else if (data.match_count === 0) {
    $("sync-btn").disabled = false;
    setLog("Key loaded. Sync to pull Faker's latest ranked solo games.");
  } else {
    $("sync-btn").disabled = false;
  }
  return data;
}

async function loadMatches() {
  const qs = selectedPuuid ? `?puuid=${encodeURIComponent(selectedPuuid)}` : "";
  const { matches } = await (await fetch(`/api/matches${qs}`)).json();
  const root = $("match-list");
  if (!matches.length) {
    root.innerHTML = `<p class="empty">No games stored yet.</p>`;
    return;
  }
  root.innerHTML = matches
    .map((m) => {
      const parts = m.participants || (m.participants_json ? JSON.parse(m.participants_json) : []);
      const blue = parts.filter((p) => p.team_id === 100);
      const red = parts.filter((p) => p.team_id === 200);
      const filterNote = m.champion_name
        ? `<div class="filter-note">${selectedName} was ${m.champion_name}${m.win == null ? "" : m.win ? " · win" : " · loss"}</div>`
        : "";
      return `
      <button class="match-card" data-id="${m.match_id}">
        <div class="tiny-row">${blue.map((p) => `<img class="tiny-champ" src="${champImg(p.champion_name)}" alt="" />`).join("")}</div>
        <div>
          <strong>Ranked solo</strong>
          <div>${m.patch} · ${fmtDate(m.game_creation)}${m.shoppers ? ` · ${m.shoppers} buyers` : ""}</div>
          ${filterNote}
        </div>
        <div class="tiny-row">${red.map((p) => `<img class="tiny-champ" src="${champImg(p.champion_name)}" alt="" />`).join("")}</div>
      </button>`;
    })
    .join("");
  root.querySelectorAll(".match-card").forEach((el) => {
    el.addEventListener("click", () => openMatch(el.dataset.id, el));
  });
}

function sideIcons(players, teamId) {
  return players
    .filter((p) => p.team_id === teamId)
    .map(
      (p) =>
        `<img class="tiny-champ" title="${p.champion_name} · ${p.riot_id}" src="${champImg(p.champion_name)}" alt="${p.champion_name}" />`
    )
    .join("");
}

function champBoard(board) {
  if (!board || !board.length) return "";
  const team = (teamId, label) => {
    const rows = board
      .filter((p) => p.team_id === teamId)
      .map(
        (p) => `
        <div class="item-row ${p.is_self ? "self" : ""}">
          <img class="tiny-champ" src="${champImg(p.champion_name)}" alt="" />
          <b>${p.champion_name}${p.is_self ? " · buying" : ""}</b>
          <span>${p.kills}/${p.deaths}/${p.assists}</span>
          ${itemStrip(p.items)}
        </div>`
      )
      .join("");
    return `<div><span>${label}</span>${rows}</div>`;
  };
  return `
    <div class="enemy-now">
      <span>Every champion’s items before this visit</span>
      <div class="board-teams">
        ${team(100, "Blue")}
        ${team(200, "Red")}
      </div>
    </div>`;
}

function visibleItems(items) {
  return (items || []).filter((it) => !(it && it.skip));
}

function itemStrip(items) {
  const vis = visibleItems(items);
  if (!vis.length) return `<span>empty</span>`;
  return vis
    .map((it) => {
      const id = it.item_id ?? it;
      const name = it.item_name || id;
      return `<img class="item-icon" title="${name}" src="${itemImg(id)}" alt="${name}" />`;
    })
    .join("");
}

async function openMatch(id, card) {
  document.querySelectorAll(".match-card").forEach((el) => el.classList.toggle("on", el === card));
  const match = await (await fetch(`/api/matches/${id}`)).json();
  const parts = match.participants || [];
  const blue = parts.filter((p) => p.team_id === 100);
  const red = parts.filter((p) => p.team_id === 200);
  const events = (match.events || []).filter((e) => e.type === "shop");
  const shoppers = match.shoppers || new Set(events.map((e) => e.puuid).filter(Boolean)).size;
  const partial =
    shoppers && shoppers < parts.length
      ? `<p class="empty">Only ${shoppers} of ${parts.length} players have shops stored. This game was collected from a single account, so it still looks like one champion’s story.</p>`
      : "";
  const cards = events
    .map((e, idx) => {
      const bought = itemStrip(e.bought);
      const time =
        e.ts === (e.ts_end ?? e.ts) ? fmtTime(e.ts) : `${fmtTime(e.ts)}–${fmtTime(e.ts_end)}`;
      return `
        ${idx ? `<div class="timeline-arrow" aria-hidden="true">↓</div>` : ""}
        <div class="decision visit-card">
          <div class="decision-head">
            <div>
              <strong>${e.champion_name || "Shop"} buys</strong>
              <div class="badge done">${e.riot_id || e.puuid || ""} · ${e.team_id === 100 ? "Blue" : "Red"}</div>
            </div>
            <span class="time">${time}</span>
          </div>
          <div class="meta-row">
            <span>Gold ~ ${e.gold ?? "?"}</span>
            <span>Lv ${e.level ?? "?"}</span>
            <span>CS ${e.cs ?? "?"}</span>
            <span>KDA ${e.kills}/${e.deaths}/${e.assists}</span>
          </div>
          <div class="build-now">
            <span>This visit</span>
            <div class="item-row">${bought}</div>
          </div>
          <div class="build-now">
            <span>Build after</span>
            <div class="item-row">${itemStrip(e.inventory_after)}</div>
          </div>
          ${champBoard(e.board)}
        </div>`;
    })
    .join("");

  $("match-detail").classList.remove("empty");
  $("match-detail").innerHTML = `
    <h2>Game timeline · ${match.patch}</h2>
    <p class="empty">${match.match_id} · ${fmtDate(match.game_creation)} · ${events.length} shops · ${shoppers || "?"} buyers</p>
    ${partial}
    <div class="teams">
      <div class="team">${sideIcons(blue, 100)} <span>Blue${match.win_team === 100 ? " · win" : ""}</span></div>
      <div class="team">${sideIcons(red, 200)} <span>Red${match.win_team === 200 ? " · win" : ""}</span></div>
    </div>
    <div class="timeline">${cards || "<p>No events stored for this game.</p>"}</div>
  `;
}

async function syncFaker() {
  const count = Number($("sync-count").value) || 15;
  $("sync-btn").disabled = true;
  setLog("Starting sync…");
  try {
    const res = await fetch(`/api/sync?count=${count}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Sync failed");
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const chunks = buf.split("\n\n");
      buf = chunks.pop() || "";
      for (const chunk of chunks) {
        const line = chunk.split("\n").find((row) => row.startsWith("data: "));
        if (!line) continue;
        const data = JSON.parse(line.slice(6));
        setLog(data.message || data.step);
      }
    }
  } catch (err) {
    setLog(String(err.message || err));
  } finally {
    $("sync-btn").disabled = false;
    await loadAll();
  }
}

async function selectPlayer(player) {
  selectedPuuid = player.puuid;
  localStorage.setItem("bt-puuid", selectedPuuid);
  renderChip(player, player.games);
  $("player-results").hidden = true;
  $("player-search").value = "";
  $("match-detail").classList.add("empty");
  $("match-detail").innerHTML =
    "<p>Filtered to this player’s games. Open one for the whole lobby’s shops, not just theirs.</p>";
  await loadAll();
}

async function clearPlayer() {
  selectedPuuid = "";
  selectedName = "this player";
  localStorage.removeItem("bt-puuid");
  renderChip(null);
  $("match-detail").classList.add("empty");
  $("match-detail").innerHTML =
    "<p>Pick a game. The timeline is the whole lobby: whoever bought next, in order.</p>";
  await loadAll();
}

async function searchPlayers(query) {
  const { players } = await (await fetch(`/api/players?q=${encodeURIComponent(query)}`)).json();
  const root = $("player-results");
  if (!root) return players;
  if (!players.length) {
    root.hidden = false;
    root.innerHTML = `<p class="empty" style="padding:10px 12px">No stored perspectives for that name.</p>`;
    return players;
  }
  root.hidden = false;
  root.innerHTML = players
    .map(
      (p) => `
      <button type="button" data-puuid="${p.puuid}">
        <span>${p.game_name}#${p.tag_line}</span>
        <span>${p.games}</span>
      </button>`
    )
    .join("");
  root.querySelectorAll("button").forEach((el) => {
    el.addEventListener("click", () => {
      const player = players.find((p) => p.puuid === el.dataset.puuid);
      if (player) selectPlayer(player);
    });
  });
  return players;
}

async function loadAll() {
  await loadStatus();
  await loadMatches();
}

$("sync-btn").addEventListener("click", syncFaker);
$("player-chip")?.addEventListener("click", () => {
  if (selectedPuuid) clearPlayer();
});
const search = $("player-search");
let searchTimer;
search.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => searchPlayers(search.value.trim()), 180);
});
search.addEventListener("focus", () => searchPlayers(search.value.trim()));
document.addEventListener("click", (ev) => {
  if (!$("player-picker")?.contains(ev.target) && !ev.target.closest?.(".player-picker")) {
    const box = $("player-results");
    if (box) box.hidden = true;
  }
});

(async () => {
  try {
    await loadAll();
    setLog("Showing every stored game. Search a player only to find their matches.");
  } catch (err) {
    setLog(String(err));
  }
})();
