const chat = document.querySelector("#chat");
const sources = document.querySelector("#sources");
const form = document.querySelector("#chatForm");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#sendButton");
const template = document.querySelector("#messageTemplate");
const topK = document.querySelector("#topK");
const topKValue = document.querySelector("#topKValue");
const threshold = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#thresholdValue");
const charCount = document.querySelector("#charCount");
const clearChat = document.querySelector("#clearChat");
const exactPhrase = document.querySelector("#exactPhrase");
const evalLimit = document.querySelector("#evalLimit");
const evalLimitValue = document.querySelector("#evalLimitValue");
const runEval = document.querySelector("#runEval");
const evalSummary = document.querySelector("#evalSummary");
const STORAGE_KEY = "rag-chat-messages-v2";
const SESSION_ID = "html-ui";

const WELCOME_TEXT =
  "Xin chào! Hãy đặt câu hỏi về pháp luật phòng, chống ma túy. Tôi sẽ truy xuất tài liệu và trả lời kèm trích dẫn nguồn.";

let messages = loadMessages();
let latestSources = getLatestSources(messages);

function loadMessages() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

function saveMessages() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

function getLatestSources(items) {
  const lastWithSources = [...items].reverse().find((item) => item.sources?.length);
  return lastWithSources?.sources || [];
}

function escapeHtml(value) {
  return fixText(value)
    .normalize("NFC")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fixText(value) {
  const text = String(value ?? "");
  if (!/[ÃÄÂáºá»Æ]/.test(text)) return text.normalize("NFC");
  try {
    return decodeURIComponent(escape(text)).normalize("NFC");
  } catch {
    return text.normalize("NFC");
  }
}

function renderCitations(text) {
  return escapeHtml(text).replace(/(\[[^\[\]]{2,120}\])/g, '<span class="citation">$1</span>');
}

function currentTime() {
  return new Date().toLocaleTimeString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderMessage(message) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(message.role);
  node.querySelector(".content").innerHTML =
    message.role === "assistant" ? renderCitations(message.content) : escapeHtml(message.content);
  node.querySelector(".time").textContent = message.time || "10:15";
  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;
}

function shortDescription(source) {
  const content = fixText(source.preview || source.content || "");
  return content.replace(/\s+/g, " ").trim().slice(0, 120);
}

function articleLabel(source) {
  // Derive a real label from the actual chunk content, not a hardcoded guess.
  const content = fixText(source.content || source.preview || "");
  const dieu = content.match(/Điều\s+(\d+[a-z]?)/i);
  if (dieu) return `Điều ${dieu[1]}`;
  const chuong = content.match(/Chương\s+([IVXLCDM]+|\d+)/i);
  if (chuong) return `Chương ${chuong[1]}`;
  const path = source.source_path || source.metadata?.source_path || "";
  const fromPath = path.match(/dieu[-_\s]*(\d+)/i);
  if (fromPath) return `Điều ${fromPath[1]}`;
  const type = source.type || source.metadata?.type;
  if (type === "news") return "Bản tin";
  const idx = source.chunk_index;
  return Number.isInteger(idx) ? `Trích đoạn #${idx}` : "Trích đoạn";
}

function renderSources(items) {
  if (!items.length) {
    sources.innerHTML =
      '<p class="sources-empty">Chưa có nguồn — hãy đặt một câu hỏi để xem tài liệu trích dẫn.</p>';
    return;
  }
  const cards = items.slice(0, 3);
  sources.innerHTML = cards
    .map((source) => {
      const metadata = source.metadata || {};
      const title = fixText(source.citation || source.source || metadata.source || metadata.title || "Source document");
      const score = Math.min(1, Number(source.score || 0)).toFixed(3);
      return `
        <article class="source-card">
          <div class="pdf-icon">PDF</div>
          <div>
            <div class="source-title">${escapeHtml(title)}</div>
            <div class="source-desc">${escapeHtml(shortDescription(source))}</div>
          </div>
          <div class="source-footer">
            <span class="score-pill">score ${score}</span>
            <span class="article-link">${escapeHtml(articleLabel(source))} ↗</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderWelcome() {
  renderMessage({ role: "assistant", content: WELCOME_TEXT, time: currentTime() });
}

function renderAll() {
  chat.innerHTML = "";
  if (messages.length === 0) {
    renderWelcome();
  } else {
    messages.forEach(renderMessage);
  }
  latestSources = getLatestSources(messages);
  renderSources(latestSources);
}

async function ask(question) {
  const userMessage = { role: "user", content: question, time: currentTime() };
  messages.push(userMessage);
  saveMessages();
  renderMessage(userMessage);

  sendButton.disabled = true;
  questionInput.disabled = true;

  const loading = {
    role: "assistant",
    content: "Đang truy xuất tài liệu bằng hybrid retrieval và sinh câu trả lời có citation...",
    time: currentTime(),
  };
  renderMessage(loading);
  const loadingNode = chat.lastElementChild;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        session_id: SESSION_ID,
        top_k: Number(topK.value),
        score_threshold: Number(threshold.value),
        exact_phrase: Boolean(exactPhrase?.checked),
      }),
    });
    const result = await response.json();
    const assistantMessage = {
      role: "assistant",
      content: result.answer,
      sources: result.sources || [],
      retrieval_source: result.retrieval_source || "hybrid",
      time: currentTime(),
    };
    messages.push(assistantMessage);
    latestSources = assistantMessage.sources.length ? assistantMessage.sources : latestSources;
    saveMessages();
    loadingNode.remove();
    renderMessage(assistantMessage);
    renderSources(latestSources);
  } catch (error) {
    const assistantMessage = {
      role: "assistant",
      content: `Chưa kết nối được backend HTML. Chi tiết: ${error.message}`,
      sources: [],
      retrieval_source: "error",
      time: currentTime(),
    };
    messages.push(assistantMessage);
    saveMessages();
    loadingNode.remove();
    renderMessage(assistantMessage);
  } finally {
    sendButton.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  charCount.textContent = "0/2000";
  ask(question);
});

questionInput.addEventListener("input", () => {
  charCount.textContent = `${questionInput.value.length}/2000`;
});

topK.addEventListener("input", () => {
  topKValue.textContent = topK.value;
});

threshold.addEventListener("input", () => {
  thresholdValue.textContent = Number(threshold.value).toFixed(2);
});

evalLimit?.addEventListener("input", () => {
  evalLimitValue.textContent = evalLimit.value;
});

runEval?.addEventListener("click", async () => {
  runEval.disabled = true;
  evalSummary.innerHTML = "Đang chạy evaluation...";
  try {
    const response = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        limit: Number(evalLimit.value),
        top_k: Number(topK.value),
        exact_phrase: Boolean(exactPhrase?.checked),
      }),
    });
    const result = await response.json();
    if (!response.ok || result.ok === false) {
      throw new Error(result.error || "Evaluation failed");
    }
    evalSummary.innerHTML = `
      <strong>Overall: ${Number(result.averages.overall).toFixed(3)}</strong><br>
      Faith: ${Number(result.averages.faithfulness).toFixed(3)} ·
      Relev: ${Number(result.averages.answer_relevancy).toFixed(3)}<br>
      Recall: ${Number(result.averages.contextual_recall).toFixed(3)} ·
      Precision: ${Number(result.averages.contextual_precision).toFixed(3)}<br>
      ${result.evaluated}/${result.dataset_size} câu · ${escapeHtml(result.judge)}
    `;
  } catch (error) {
    evalSummary.textContent = `Evaluation lỗi: ${error.message}`;
  } finally {
    runEval.disabled = false;
  }
});

clearChat.addEventListener("click", () => {
  localStorage.removeItem("rag-chat-messages");
  localStorage.removeItem(STORAGE_KEY);
  messages = [];
  latestSources = [];
  saveMessages();
  renderAll();
  fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION_ID }),
  }).catch(() => {});
});

renderAll();
