// Same-origin by default (the backend serves this UI). To host the frontend
// separately, set window.LIBRARIAN_API_BASE = "https://your-backend" before this script.
const API_BASE = (window.LIBRARIAN_API_BASE || "").replace(/\/$/, "");

const $ = (sel, root = document) => root.querySelector(sel);

const form = $("#form");
const input = $("#question");
const sendBtn = $("#send");
const thread = $("#thread");
const intro = $("#intro");
const statusEl = $("#status");

// ---------- Theme ----------
const savedTheme = localStorage.getItem("theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
$("#theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("theme", next);
});

// ---------- Backend status ----------
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.status === "healthy") {
      statusEl.textContent = `${data.chunks_loaded.toLocaleString()} chunks · ${data.generation_model}`;
      statusEl.className = "status ok";
    } else {
      statusEl.textContent = "loading index…";
      statusEl.className = "status";
      setTimeout(checkHealth, 2000);
    }
  } catch {
    statusEl.textContent = "backend offline";
    statusEl.className = "status err";
  }
}
checkHealth();

// ---------- Helpers ----------
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function fmtMs(ms) {
  return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms";
}

// Small safe renderer: code blocks, inline code, bold, lists, paragraphs, [S#] citations.
function renderAnswer(text, nSources) {
  const blocks = [];
  let t = esc(text).replace(/```(\w*)\n?([\s\S]*?)```/g, (_, _lang, code) => {
    blocks.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });

  t = t
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[((?:S\d+)(?:\s*,\s*S\d+)*)\]/g, (match, group) => {
      const nums = group.split(",").map((s) => parseInt(s.trim().slice(1), 10));
      if (nums.some((n) => n < 1 || n > nSources)) return match;
      return nums.map((n) => `<button type="button" class="cite" data-n="${n}" title="Show source ${n}">${n}</button>`).join("");
    });

  const html = t.split(/\n{2,}/).map((para) => {
    const trimmed = para.trim();
    if (/^\u0000\d+\u0000$/.test(trimmed)) return trimmed;
    const lines = trimmed.split("\n");
    if (lines.every((l) => /^\s*[-*•] /.test(l))) {
      return "<ul>" + lines.map((l) => `<li>${l.replace(/^\s*[-*•] /, "")}</li>`).join("") + "</ul>";
    }
    if (lines.every((l) => /^\s*\d+[.)] /.test(l))) {
      return "<ol>" + lines.map((l) => `<li>${l.replace(/^\s*\d+[.)] /, "")}</li>`).join("") + "</ol>";
    }
    return `<p>${lines.join("<br>")}</p>`;
  }).join("");

  return html.replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[+i]);
}

// ---------- Rendering a turn ----------
function addTurn(question) {
  intro.hidden = true;
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="q"></div>
    <div class="card"><div class="card-inner">
      <div class="loading"><span class="dot"></span><span class="dot"></span><span class="dot"></span>
      <span>Searching the documentation…</span></div>
    </div></div>`;
  $(".q", turn).textContent = question;
  thread.appendChild(turn);
  turn.scrollIntoView({ behavior: "smooth", block: "start" });
  return turn;
}

function renderResult(turn, question, data) {
  const inner = $(".card-inner", turn);
  const n = data.sources.length;
  inner.classList.toggle("abstain", data.abstained);

  const sourcesHtml = data.sources.map((s) => `
    <div class="source" data-n="${s.id}">
      <div class="source-head">
        <span class="num">${s.id}</span>
        <a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a>
      </div>
      <p>${esc(s.snippet)}${s.snippet.length >= 500 ? "…" : ""}</p>
    </div>`).join("");

  const t = data.timings, u = data.usage;
  inner.innerHTML = `
    ${data.abstained ? '<div class="abstain-tag">Not in the documentation</div>' : ""}
    <div class="answer">${renderAnswer(data.answer, n)}</div>

    <details class="sources" ${data.abstained ? "" : "open"}>
      <summary>Sources (${n})</summary>
      <div class="source-list">${sourcesHtml}</div>
    </details>

    <details class="meta">
      <summary>Timing &amp; tokens</summary>
      <div class="meta-grid">
        <div><b>${fmtMs(t.retrieval_ms)}</b><span>Retrieval</span></div>
        <div><b>${fmtMs(t.generation_ms)}</b><span>Generation</span></div>
        <div><b>${fmtMs(t.total_ms)}</b><span>Total</span></div>
        <div><b>${u.input_tokens} / ${u.output_tokens}</b><span>Tokens in / out</span></div>
        <div><b>${esc(data.model)}</b><span>Model</span></div>
      </div>
    </details>

    <div class="actions">
      <span class="thanks">Was this helpful?</span>
      <button type="button" class="btn" data-rate="up" aria-label="Helpful">👍</button>
      <button type="button" class="btn" data-rate="down" aria-label="Not helpful">👎</button>
      <span class="spacer"></span>
      <button type="button" class="btn" data-copy>Copy answer</button>
    </div>`;

  // Citation chips -> open + highlight the source card.
  inner.addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    if (!cite) return;
    const details = $("details.sources", inner);
    details.open = true;
    const card = $(`.source[data-n="${cite.dataset.n}"]`, inner);
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1500);
  });

  // Copy.
  $("[data-copy]", inner).addEventListener("click", async (e) => {
    try {
      await navigator.clipboard.writeText(data.answer);
      e.target.textContent = "Copied ✓";
      setTimeout(() => (e.target.textContent = "Copy answer"), 1500);
    } catch { /* clipboard blocked */ }
  });

  // Feedback.
  const actions = $(".actions", inner);
  actions.querySelectorAll("[data-rate]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const rating = btn.dataset.rate;
      // Feedback endpoint not provided by backend; local visual state only.
      actions.querySelectorAll("[data-rate]").forEach((b) => (b.disabled = true));
      btn.classList.add(`active-${rating}`);
      $(".thanks", actions).textContent = "Feedback selected (not submitted)";

    });
  });
}

// ---------- Asking ----------
async function ask(question) {
  question = question.trim();
  if (!question) return;
  sendBtn.disabled = true;
  input.value = "";
  autoGrow();
  const turn = addTurn(question);

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }
    const raw = await res.json();
    const normalized = {
      answer: raw.answer,
      abstained: /i don.t know based on the available python documentation/i.test(raw.answer),
      sources: (raw.sources || []).map((source, index) => ({
        id: index + 1,
        title: source.title,
        url: source.url,
        snippet: ""
      })),
      timings: raw.timing,
      usage: raw.token_usage,
      model: "The Librarian AI"
    };
    renderResult(turn, question, normalized);
  } catch (err) {
    $(".card-inner", turn).innerHTML =
      `<p class="error"><strong>Something went wrong.</strong> ${esc(err.message)}</p>`;
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  ask(input.value);
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}
input.addEventListener("input", autoGrow);

document.querySelectorAll("#examples .chip").forEach((chip) =>
  chip.addEventListener("click", () => ask(chip.textContent))
);
