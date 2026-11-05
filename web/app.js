// Workbench front-end. Plain ES module, no build step.
import { icon } from "./icons.js";

const BASE = "/api/plugin/kira_plugin_aiimg/aiimg";
const THEME_KEY = "aiimg-theme";
const GALLERY_PAGE = 60;

const VIEWS = {
  overview: {
    label: "概览",
    title: "概览",
    sub: "服务商、队列与最近一段时间的结果",
    icon: "dashboard",
  },
  providers: {
    label: "服务商",
    title: "服务商",
    sub: "先配连接，再决定每个功能按什么顺序出图",
    icon: "plug",
  },
  tasks: {
    label: "任务",
    title: "任务",
    sub: "排队、运行和历史记录",
    icon: "tasks",
  },
  gallery: {
    label: "画廊",
    title: "画廊",
    sub: "生成结果留档，收藏的不会被自动清理",
    icon: "gallery",
  },
  settings: {
    label: "设置",
    title: "设置",
    sub: "权限、默认画幅、队列和框架接管",
    icon: "settings",
  },
};

const STATUS = {
  queued: ["排队中", "wait"],
  running: ["运行中", "run"],
  succeeded: ["已完成", "ok"],
  failed: ["失败", "fail"],
  cancelled: ["已取消", "off"],
};

const KIND_LABELS = {
  generate_image: "文生图",
  edit_image: "改图",
  generate_selfie: "自拍",
  generate_image_batch: "批量生图",
  generate_video: "视频",
};

const state = {
  view: "overview",
  config: null,
  original: null,
  revision: "",
  sections: {},
  templates: [],
  fields: {},
  features: [],
  providerDefaults: {},
  required: {},
  status: {},
  notes: [],
  providerTab: "connections",
  galleryTab: "all",
  selected: null,
  providerError: "",
  tasks: [],
  gallery: [],
  galleryTotal: 0,
  galleryPage: 1,
  galleryQuery: "",
  galleryKind: "",
  gallerySort: "new",
  gallerySelect: false,
  galleryPicked: [],
  galleryTimer: 0,
  taskState: "",
  refFiles: [],
  metrics: [],
  saving: false,
  polling: 0,
  loadError: "",
  // Read-only list on the settings page. It is fetched once per visit, kept
  // here so the card can redraw itself without asking again.
  inspirations: { items: [], total: 0, loaded: false, loading: false, error: "" },
  // True until the very first load() settles: while it is on, a missing
  // config means "still reading", not "reading failed".
  booting: true,
  startupDismissed: "",
};

const $ = (selector, root = document) => root.querySelector(selector);

function esc(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch])
  );
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function iconNode(name, cls) {
  return icon(name, cls);
}

async function api(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    credentials: "same-origin",
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let data = null;
  if ((response.headers.get("content-type") || "").includes("application/json")) {
    data = await response.json().catch(() => null);
  }
  if (!response.ok) {
    const detail = data && (data.detail || data.error || data.message);
    const text =
      typeof detail === "string"
        ? detail
        : detail
        ? JSON.stringify(detail)
        : `请求没有成功（${response.status}）`;
    throw new Error(text);
  }
  return data || {};
}

let toastTimer = 0;
function toast(message, tone = "info") {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${tone === "error" ? " error" : ""}`;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    node.className = "toast";
  }, tone === "error" ? 6000 : 2600);
}

// -- data -------------------------------------------------------------

// One response carries everything the form needs; only `status` and `notes`
// touch the gallery, the queue and the provider ledger, so they can arrive
// later without holding the first paint hostage.
function applyConfig(data) {
  state.config = data.config || {};
  state.original = structuredClone(state.config);
  state.revision = data.revision || "";
  state.sections = data.sections || {};
  state.templates = data.templates || [];
  state.fields = data.fields || {};
  state.features = data.features || [];
  state.providerDefaults = data.provider_defaults || {};
  state.required = data.required || {};
  state.status = data.status || {};
  state.notes = data.notes || [];
  state.providerError = "";
  state.loadError = "";
  const providers = state.config.providers || [];
  if (!providers.length) state.selected = null;
  else if (!providers.some((item) => item.id === state.selected)) {
    state.selected = providers[0].id;
  }
}

async function load() {
  // Paint the form from the light payload, then fill in the live numbers and
  // the reference library side by side. A broken status block therefore costs
  // a number, not the whole page.
  applyConfig(await api("/config?lite=1"));
  render();
  await Promise.all([loadReferenceFiles(), loadMetrics(), loadStatus()]);
  render();
}

async function loadStatus() {
  try {
    const data = await api("/config");
    state.status = data.status || {};
    state.notes = data.notes || [];
  } catch (error) {
    state.status = state.status || {};
  }
}

// Health numbers come from the plugin's own ledger, so a failed request only
// means the rows fall back to "no history yet".
async function loadMetrics() {
  try {
    const data = await api("/metrics");
    state.metrics = data.providers || [];
  } catch (error) {
    state.metrics = [];
  }
}

// Reference photos are not part of the config, so the list is read from the
// plugin data directory instead of the settings payload.
async function loadReferenceFiles() {
  try {
    const data = await api("/files");
    state.refFiles = normalizeReferenceFiles(data.files);
  } catch (error) {
    state.refFiles = [];
  }
}

function normalizeReferenceFiles(files) {
  return (files || [])
    .map((item) => ({
      name: String(item?.name || ""),
      role: String(item?.role || "subject"),
      size: Number(item?.size || 0),
      exists: item?.exists !== false,
    }))
    .filter((item) => item.name);
}

const SCOPE_LABELS = {
  providers: "保存服务商",
  chains: "保存链路",
  settings: "保存设置",
};

function currentScope() {
  if (state.view === "providers") {
    return state.providerTab === "chains" ? "chains" : "providers";
  }
  if (state.view === "settings") return "settings";
  return "";
}

function sectionKeys() {
  return Object.keys(state.sections || {}).filter((key) => key.startsWith("section_"));
}

function scopePayload(config, scope) {
  const source = config || {};
  if (scope === "providers") return { providers: source.providers || [] };
  if (scope === "chains") return { chains: source.chains || {} };
  const payload = {};
  for (const key of sectionKeys()) payload[key] = source[key] || {};
  return payload;
}

function dirty(scope = currentScope()) {
  if (!scope || !state.config || !state.original) return false;
  return (
    JSON.stringify(scopePayload(state.config, scope)) !==
    JSON.stringify(scopePayload(state.original, scope))
  );
}

async function save(allowReload = false) {
  const scope = currentScope();
  if (!scope || state.saving) return false;
  state.saving = true;
  updateTopbar();
  try {
    const data = await api("/config", {
      method: "PUT",
      body: JSON.stringify({
        revision: state.revision,
        config: scopePayload(state.config, scope),
        allow_reload: allowReload,
      }),
    });
    const saved = data.config || state.config;
    const keep = {};
    for (const key of ["providers", "chains", ...sectionKeys()]) {
      if (!(key in scopePayload(saved, scope))) keep[key] = state.config[key];
    }
    state.config = { ...saved, ...keep };
    state.original = structuredClone(saved);
    state.revision = data.revision || state.revision;
    state.status = data.status || state.status;
    state.notes = data.notes || [];
    state.providerError = "";
    toast("已保存并生效");
    render();
    return true;
  } catch (error) {
    const message = String(error.message || error);
    if (!allowReload && message.includes("任务在跑") && window.confirm(`${message}。现在就应用吗？`)) {
      state.saving = false;
      return save(true);
    }
    toast(message, "error");
    return false;
  } finally {
    state.saving = false;
    updateTopbar();
  }
}

// -- chrome -----------------------------------------------------------

function renderNav() {
  const nav = $("#nav");
  nav.replaceChildren();
  const providers = (state.config?.providers || []).length;
  const active = Number(state.status?.queue?.running || 0) + Number(state.status?.queue?.waiting || 0);
  for (const [key, view] of Object.entries(VIEWS)) {
    const count =
      key === "providers" ? providers : key === "tasks" ? active : key === "gallery" ? (state.status?.gallery?.total || 0) : 0;
    const button = el(
      "button",
      {
        class: "nav-item",
        type: "button",
        "data-nav": key,
        "aria-current": state.view === key ? "page" : false,
      },
      iconNode(view.icon),
      el("span", {}, view.label),
      count ? el("span", { class: "count" }, String(count)) : null
    );
    nav.append(button);
  }
  const version = $("#version");
  version.textContent = state.status?.version ? `v${state.status.version}` : "";
}

function updateTopbar() {
  const view = VIEWS[state.view];
  $("#view-title").textContent = view.title;
  $("#view-sub").textContent = view.sub;
  const scope = currentScope();
  const changed = Boolean(scope) && dirty(scope);
  $("#dirty").classList.toggle("hidden", !changed);
  $("#btn-save").classList.toggle("hidden", !changed);
  $("#btn-discard").classList.toggle("hidden", !changed);
  const saveButton = $("#btn-save");
  saveButton.disabled = state.saving;
  saveButton.replaceChildren(
    iconNode("save"),
    document.createTextNode(state.saving ? "保存中" : SCOPE_LABELS[scope] || "保存")
  );
}

function renderNotes() {
  const box = $("#notes");
  box.replaceChildren();
  const notes = [...(state.status?.provider_errors || []), ...(state.notes || [])];
  if (state.providerError) notes.unshift(state.providerError);
  for (const note of notes) {
    box.append(
      el("div", { class: "note warn" }, iconNode("alert"), el("span", {}, String(note)))
    );
  }
  box.classList.toggle("hidden", notes.length === 0);
}

// A plugin that failed to start still serves the page, it just cannot do part
// of its job. That is said once at the top of the page, in a line the reader
// can close, not as a wall that replaces the whole screen.
function renderStartupBanner() {
  const main = $(".main");
  if (!main) return;
  const text = String(state.status?.startup_error || "").trim();
  let box = $("#startup-banner");
  if (!text || text === state.startupDismissed) {
    if (box) box.remove();
    return;
  }
  if (!box) {
    box = el("div", { id: "startup-banner", class: "startup-banner", role: "status" });
    box.addEventListener("click", (event) => {
      if (!event.target.closest("[data-action='startup-dismiss']")) return;
      state.startupDismissed = box.dataset.text || "";
      box.remove();
    });
    main.prepend(box);
  }
  // The text is the whole content, so an unchanged message needs no repaint.
  if (box.dataset.text === text) return;
  box.dataset.text = text;
  box.replaceChildren(
    iconNode("alert"),
    el(
      "div",
      { class: "startup-text" },
      el("strong", {}, "插件启动时出错了，部分功能不可用："),
      el("span", { class: "startup-detail" }, text)
    ),
    el(
      "button",
      {
        class: "icon-btn startup-close",
        type: "button",
        title: "关闭提醒",
        "data-action": "startup-dismiss",
      },
      iconNode("close")
    )
  );
}

function render() {
  renderNav();
  updateTopbar();
  renderNotes();
  renderStartupBanner();
  const view = $("#view");
  view.replaceChildren();
  stopPolling();
  releaseBlobs();
  if (!state.config) {
    // An empty config is only an error once the first load has really failed.
    // Before that the page is just waiting, so it paints a quiet skeleton.
    if (state.booting) renderSkeleton(view);
    else renderLoadError(view);
    return;
  }
  const renderer = {
    overview: renderOverview,
    providers: renderProviders,
    tasks: renderTasks,
    gallery: renderGallery,
    settings: renderSettings,
  }[state.view];
  renderer(view);
  fitTextareas(view);
}

function releaseBlobs() {
}

// First paint: card-shaped bars in the place the real page will take. It never
// claims anything, so a slow plugin reads as "loading", not as "broken".
function renderSkeleton(root) {
  const head = el(
    "div",
    { class: "card sk-card" },
    el("div", { class: "sk-line w30" }),
    el("div", { class: "sk-line w60" }),
    el("div", { class: "sk-line w80" })
  );
  const stats = el("div", { class: "grid-stats" });
  for (let index = 0; index < 3; index += 1) {
    stats.append(
      el(
        "div",
        { class: "card sk-card" },
        el("div", { class: "sk-line w40" }),
        el("div", { class: "sk-line w70" }),
        el("div", { class: "sk-line w50" })
      )
    );
  }
  root.append(stats, head);
  root.append(el("p", { class: "sk-hint", role: "status" }, "正在读取配置…"));
}

function renderLoadError(root) {
  root.append(
    el(
      "div",
      { class: "card" },
      el(
        "div",
        { class: "blank" },
        iconNode("alert"),
        el("strong", {}, "配置没有加载出来"),
        el("p", {}, state.loadError || "网络或插件没回应，稍等一下再试。"),
        el("button", { class: "btn small", type: "button", "data-action": "boot-retry" }, "再试一次")
      )
    )
  );
}

// -- overview ---------------------------------------------------------

function statCard(iconName, label, value, foot) {
  return el(
    "div",
    { class: "card stat" },
    el("div", { class: "label" }, iconNode(iconName), el("span", {}, label)),
    el("div", { class: "value", html: value }),
    el("div", { class: "foot" }, foot || "")
  );
}

function renderOverview(root) {
  const queue = state.status?.queue || {};
  const gallery = state.status?.gallery || {};
  const providers = state.config.providers || [];
  const ready = state.status?.ready ? "正常" : "未就绪";

  const stats = el("div", { class: "grid-stats" });
  stats.append(
    statCard(
      "plug",
      "服务商",
      `${providers.length}<small>家</small>`,
      `生图 ${(state.status?.image_providers || []).length} · 视频 ${(state.status?.video_providers || []).length}`
    ),
    statCard(
      "zap",
      "队列",
      `${queue.running || 0}<small>运行 / ${queue.waiting || 0} 等待</small>`,
      `并行上限 ${queue.max_running ?? "-"} · 排队上限 ${queue.max_queued ?? "-"}`
    ),
    statCard(
      "gallery",
      "画廊",
      `${gallery.total || 0}<small>张</small>`,
      `收藏 ${gallery.favorites || 0} 张`
    ),
    statCard("info", "运行状态", ready, `版本 ${esc(state.status?.version || "-")}`)
  );

  const providerCard = el("div", { class: "card" });
  providerCard.append(
    el(
      "div",
      { class: "card-head" },
      el("h2", {}, "服务商"),
      el("span", { class: "sub" }, "点一行去编辑"),
      el("span", { class: "spacer" }),
      el(
        "button",
        { class: "btn small ghost", type: "button", "data-action": "goto", "data-view": "providers" },
        "打开服务商",
        iconNode("arrow")
      )
    )
  );
  if (!providers.length) {
    providerCard.append(
      el(
        "div",
        { class: "blank" },
        iconNode("plug"),
        el("strong", {}, "还没有服务商"),
        el("p", {}, "先去服务商页面添加一家，然后保存")
      )
    );
  } else {
    const body = el("div", {});
    for (const provider of providers) {
      const template = state.templates.find((item) => item.key === provider.template_key);
      body.append(providerRow(provider, template, () => {
        state.view = "providers";
        state.providerTab = "connections";
        state.selected = provider.id;
        location.hash = "#providers";
        render();
      }));
    }
    providerCard.append(body);
  }

  const tasksCard = el("div", { class: "card" });
  tasksCard.append(
    el(
      "div",
      { class: "card-head" },
      el("h2", {}, "最近任务"),
      el("span", { class: "spacer" }),
      el(
        "button",
        { class: "btn small ghost", type: "button", "data-action": "goto", "data-view": "tasks" },
        "全部任务"
      )
    )
  );
  tasksCard.append(el("div", { id: "overview-tasks", class: "empty" }, "读取中…"));

  root.append(stats, providerCard, tasksCard);
  loadRecentTasks();
}

async function loadRecentTasks() {
  const box = $("#overview-tasks");
  if (!box) return;
  try {
    const data = await api("/tasks?limit=6");
    const tasks = data.tasks || [];
    box.replaceChildren();
    if (!tasks.length) {
      box.className = "empty";
      box.textContent = "还没有任务记录";
      return;
    }
    box.className = "";
    box.append(taskTable(tasks, false));
  } catch (error) {
    box.textContent = String(error.message || error);
  }
}

// -- providers --------------------------------------------------------

function providerRow(provider, template, onClick, highlight = false) {
  const ok = providerComplete(provider);
  const health = providerHealth(provider.id);
  const cooling = (health?.cooling_seconds || 0) > 0;
  const detail = `${template?.label || provider.template_key || "未选类型"} · ${provider.model || "没填模型"}`;
  return el(
    "button",
    {
      class: "provider-row",
      type: "button",
      "data-selected": highlight && provider.id === state.selected ? "true" : false,
      onclick: onClick,
    },
    el("span", { class: `dot ${cooling ? "warn" : ok ? "ok" : "warn"}` }),
    el(
      "span",
      { class: "meta" },
      el("span", { class: "name" }, provider.id || "未命名"),
      el(
        "span",
        { class: "detail" },
        detail,
        healthText(health) ? el("span", { class: "health" }, healthText(health)) : null
      )
    )
  );
}

function providerHealth(providerId) {
  return (state.metrics || []).find((row) => row.id === providerId) || null;
}

function healthText(row) {
  if (!row) return "";
  if ((row.cooling_seconds || 0) > 0) return `冷却 ${Math.ceil(row.cooling_seconds)} 秒`;
  if (row.success_rate === null || row.success_rate === undefined) return "";
  const percent = Math.round(row.success_rate * 100);
  const median = row.p50_ms ? `${(row.p50_ms / 1000).toFixed(1)} 秒` : "-";
  return `成功率 ${percent}% · 中位 ${median}`;
}

function providerComplete(provider) {
  const required = state.required[provider.template_key] || [];
  return required.every((field) => {
    const value = provider[field];
    return Array.isArray(value) ? value.length > 0 : String(value || "").trim().length > 0;
  });
}

function renderProviders(root) {
  const tabs = el(
    "div",
    { class: "tabs" },
    el(
      "button",
      {
        type: "button",
        "data-action": "provider-tab",
        "data-tab": "connections",
        "aria-selected": state.providerTab === "connections" ? "true" : "false",
      },
      "连接"
    ),
    el(
      "button",
      {
        type: "button",
        "data-action": "provider-tab",
        "data-tab": "chains",
        "aria-selected": state.providerTab === "chains" ? "true" : "false",
      },
      "功能链路"
    )
  );
  root.append(tabs);
  if (state.providerTab === "chains") renderChains(root);
  else renderConnections(root);
}

function renderConnections(root) {
  const providers = state.config.providers || [];
  const split = el("div", { class: "split" });

  const list = el("div", { class: "card panel-list" });
  list.append(
    el(
      "div",
      { class: "list-head" },
      el("strong", { style: "font-size:12px" }, `连接（${providers.length}）`),
      el("span", { class: "spacer", style: "margin-left:auto" }),
      el(
        "button",
        { class: "btn small", type: "button", "data-action": "provider-add" },
        iconNode("plus"),
        "新增"
      )
    )
  );
  if (!providers.length) {
    list.append(el("div", { class: "empty" }, "还没有服务商，点上面的新增开始"));
  } else {
    for (const provider of providers) {
      const template = state.templates.find((item) => item.key === provider.template_key);
      list.append(
        providerRow(provider, template, () => {
          state.selected = provider.id;
          state.providerError = "";
          render();
        }, true)
      );
    }
  }
  split.append(list, renderProviderEditor());
  root.append(split);
}

function fieldMeta(provider, field) {
  const base = state.fields[field] || { label: field, type: "text" };
  const template = (state.templates || []).find((item) => item.key === provider.template_key);
  const override = (template?.field_overrides || {})[field] || {};
  return { ...base, ...override };
}

function fieldLabel(name) {
  return state.fields[name]?.label || name;
}

function providerInput(provider, field) {
  if (field === "api_keys") return providerKeysField(provider);
  const meta = fieldMeta(provider, field);
  const value = provider[field];
  const wrap = el("div", { class: `field${["textarea", "json", "list"].includes(meta.type) ? " wide" : ""}` });
  const required = (state.required[provider.template_key] || []).includes(field);
  workaround: {
    wrap.append(
      el(
        "label",
        {},
        meta.label || fieldLabel(field),
        required ? el("span", { class: "req" }, "必填") : null
      )
    );
  }
  let control;
  if (meta.type === "switch") {
    control = el(
      "label",
      { class: "switch" },
      el("input", {
        type: "checkbox",
        checked: provider[field] === true,
        "data-provider-field": field,
        "data-kind": "bool",
      }),
      el("span", { class: "track" }),
      el("span", {}, provider[field] === true ? "已开启" : "已关闭")
    );
    control.dataset.label = provider[field] === true ? "已开启" : "已关闭";
  } else if (meta.type === "select") {
    control = el(
      "select",
      { "data-provider-field": field, "data-kind": "text" },
      ...(meta.options || []).map((option) =>
        el("option", { value: option, selected: provider[field] === option ? true : false }, option)
      )
    );
  } else if (meta.type === "integer") {
    control = el("input", {
      type: "number",
      value: provider[field] ?? "",
      min: meta.minimum ?? meta.min ?? "",
      max: meta.maximum ?? meta.max ?? "",
      "data-provider-field": field,
      "data-kind": "int",
    });
  } else if (meta.type === "json") {
    const raw = provider[field];
    const text = raw && typeof raw === "object" ? JSON.stringify(raw, null, 2) : "";
    control = el("textarea", {
      "data-provider-field": field,
      "data-kind": "json",
      spellcheck: "false",
      placeholder: meta.example || "",
    });
    control.value = text;
  } else if (meta.type === "list") {
    control = el("textarea", {
      "data-provider-field": field,
      "data-kind": "list",
      placeholder: meta.example || "",
    });
    control.value = (Array.isArray(provider[field]) ? provider[field] : []).join("\n");
  } else if (meta.type === "textarea") {
    control = el("textarea", {
      "data-provider-field": field,
      "data-kind": "text",
      placeholder: meta.example || "",
      spellcheck: "false",
    });
    control.value = provider[field] || "";
  } else {
    control = el("input", {
      type: "text",
      value: provider[field] || "",
      "data-provider-field": field,
      "data-kind": "text",
      spellcheck: "false",
    });
  }
  if (field === "model") {
    const row = el("div", { class: "model-row" });
    row.append(control);
    row.append(
      el(
        "button",
        {
          class: "btn small ghost",
          type: "button",
          "data-action": "fetch-models",
          title: "按接口地址拉取模型列表",
        },
        iconNode("refresh"),
        "获取模型"
      )
    );
    wrap.append(row);
    if (meta.hint) wrap.append(el("div", { class: "hint" }, meta.hint));
    return wrap;
  }
  wrap.append(control);
  if (meta.hint) wrap.append(el("div", { class: "hint" }, meta.hint));
  return wrap;
}

function renderProviderEditor() {
  const card = el("div", { class: "card" });
  const providers = state.config.providers || [];
  const provider = providers.find((item) => item.id === state.selected);
  if (!provider) {
    card.append(
      el(
        "div",
        { class: "blank" },
        iconNode("plug"),
        el("strong", {}, "左边选一家，或新增一家"),
        el("p", {}, "配好连接和模型，保存后就能出图")
      )
    );
    return card;
  }
  const template = state.templates.find((item) => item.key === provider.template_key) || state.templates[0];

  card.append(
    el(
      "div",
      { class: "card-head" },
      el("h2", {}, provider.id || "新服务商"),
      el("span", { class: "sub" }, template?.label || ""),
      el("span", { class: "spacer" }),
      el(
        "button",
        { class: "btn small danger", type: "button", "data-action": "provider-delete" },
        iconNode("trash"),
        "删除"
      )
    )
  );

  const form = el("div", { class: "form" });
  form.append(
    el(
      "div",
      { class: "field" },
      el("label", {}, "服务商 ID", el("span", { class: "req" }, "必填")),
      el("input", {
        type: "text",
        value: provider.id || "",
        "data-provider-id": "1",
        spellcheck: "false",
        placeholder: "main",
      }),
      el("div", { class: "hint" }, state.fields.id?.hint || "")
    ),
    el(
      "div",
      { class: "field" },
      el("label", {}, "接口类型"),
      el(
        "select",
        { "data-provider-id": "template" },
        ...state.templates.map((item) =>
          el(
            "option",
            { value: item.key, selected: item.key === provider.template_key ? true : false },
            item.label
          )
        )
      ),
      el("div", { class: "hint" }, template?.summary || "")
    )
  );
  const fields = template?.fields || [];
  const renderedRows = new Set();
  const renderedFields = new Set();
  for (const field of fields) {
    if (renderedFields.has(field)) continue;
    const meta = fieldMeta(provider, field);
    if (meta.row) {
      if (renderedRows.has(meta.row)) continue;
      const groupFields = fields.filter((item) => fieldMeta(provider, item).row === meta.row);
      groupFields.forEach((item) => renderedFields.add(item));
      renderedRows.add(meta.row);
      const row = el("div", { class: "field-row" });
      for (const item of groupFields) row.append(providerInput(provider, item));
      form.append(row);
      continue;
    }
    form.append(providerInput(provider, field));
  }
  card.append(el("div", { class: "card-body" }, form));
  return card;
}

function renderChains(root) {
  const providers = state.config.providers || [];
  const wrap = el("div", {});
  if (!providers.length) {
    wrap.append(
      el(
        "div",
        { class: "card" },
        el(
          "div",
          { class: "blank" },
          iconNode("flow"),
          el("strong", {}, "先添加服务商"),
          el("p", {}, "有了连接之后，这里才能排顺序")
        )
      )
    );
    root.append(wrap);
    return;
  }
  for (const feature of state.features) {
    const chain = state.config.chains?.[feature.key] || [];
    const card = el("div", { class: "card chain-card" });
    const body = el("div", { class: "card-body" });
    body.append(
      el(
        "div",
        { class: "chain-title" },
        el("h3", {}, feature.label || feature.key),
        el("span", { class: "hint" }, feature.hint || "")
      )
    );
    const rows = el("div", { class: "chain-rows" });
    if (!chain.length) {
      rows.append(el("div", { class: "empty" }, "没有指定顺序，会自动用第一个能用的服务商"));
    }
    chain.forEach((item, index) => {
      rows.append(
        el(
          "div",
          { class: "chain-row" },
          el("div", { class: "idx" }, String(index + 1)),
          el(
            "select",
            {
              "data-chain": feature.key,
              "data-chain-index": String(index),
              "data-chain-field": "provider",
            },
            ...providers.map((provider) =>
              el(
                "option",
                { value: provider.id, selected: provider.id === item.provider ? true : false },
                provider.id
              )
            )
          ),
          el("input", {
            type: "text",
            class: "output",
            value: item.output || "",
            placeholder: "画幅（可留空）",
            "data-chain": feature.key,
            "data-chain-index": String(index),
            "data-chain-field": "output",
          }),
          el(
            "div",
            { class: "chain-actions" },
            el(
              "button",
              {
                class: "btn ghost icon-btn small",
                type: "button",
                title: "上移",
                "data-action": "chain-move",
                "data-feature": feature.key,
                "data-index": String(index),
                "data-delta": "-1",
                disabled: index === 0,
              },
              iconNode("up")
            ),
            el(
              "button",
              {
                class: "btn ghost icon-btn small",
                type: "button",
                title: "下移",
                "data-action": "chain-move",
                "data-feature": feature.key,
                "data-index": String(index),
                "data-delta": "1",
                disabled: index === chain.length - 1,
              },
              iconNode("down")
            ),
            el(
              "button",
              {
                class: "btn ghost icon-btn small",
                type: "button",
                title: "移除",
                "data-action": "chain-remove",
                "data-feature": feature.key,
                "data-index": String(index),
              },
              iconNode("close")
            )
          )
        )
      );
    });
    body.append(rows);
    const available = providers.filter(
      (provider) => !chain.some((item) => item.provider === provider.id)
    );
    body.append(
      el(
        "div",
        { class: "chain-add" },
        el(
          "select",
          { "data-chain-add": feature.key, disabled: available.length === 0 },
          el(
            "option",
            { value: "" },
            available.length ? "添加到链路…" : "所有服务商都已在链路上"
          ),
          ...available.map((provider) => el("option", { value: provider.id }, provider.id))
        )
      )
    );
    card.append(body);
    wrap.append(card);
  }
  root.append(wrap);
}

// -- tasks ------------------------------------------------------------

const TASK_FILTERS = [
  ["", "全部"],
  ["running", "运行中"],
  ["succeeded", "已完成"],
  ["failed", "失败"],
  ["cancelled", "已取消"],
];

function statusChip(status) {
  const [label, tone] = STATUS[status] || [status || "未知", "wait"];
  return el("span", { class: `chip ${tone}` }, label);
}

function formatTime(ts) {
  if (!ts) return "-";
  const date = new Date(Number(ts) * 1000);
  if (Number.isNaN(date.getTime())) return "-";
  const pad = (value) => String(value).padStart(2, "0");
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function duration(task) {
  const start = Number(task.started_at || 0);
  const end = Number(task.finished_at || 0) || (task.status === "running" ? Date.now() / 1000 : 0);
  if (!start || !end || end < start) return "-";
  const seconds = Math.round(end - start);
  return seconds >= 60 ? `${Math.floor(seconds / 60)}分${seconds % 60}秒` : `${seconds}秒`;
}

// Delivery is read-only here: the task table reports what happened on the way
// out, it never offers to send again (the model has its own resend tool).
const DELIVERY_LABELS = {
  delivered: ["已送达", "ok"],
  failed: ["发送失败", "fail"],
};

function deliveryChip(task) {
  const entry = DELIVERY_LABELS[String(task.delivery_state || "")];
  if (!entry) return null;
  const chip = el("span", { class: `chip ${entry[1]}` }, entry[0]);
  const attempts = Number(task.delivery_attempts || 0);
  if (attempts > 0) chip.title = `投递尝试 ${attempts} 次`;
  return chip;
}

function taskTable(tasks, withActions) {
  const table = el("table", { class: "data" });
  const head = el(
    "tr",
    {},
    el("th", {}, "功能"),
    el("th", {}, "状态"),
    el("th", {}, "投递"),
    el("th", {}, "服务商"),
    el("th", {}, "提示词"),
    el("th", {}, "创建"),
    el("th", {}, "耗时"),
    withActions ? el("th", {}, "") : null
  );
  table.append(el("thead", {}, head));
  const body = el("tbody", {});
  for (const task of tasks) {
    const canCancel = task.status === "queued" || task.status === "running";
    const rowActions = [];
    if (canCancel) {
      rowActions.push(
        el(
          "button",
          {
            class: "btn small ghost",
            type: "button",
            "data-action": "task-cancel",
            "data-task": task.task_id,
          },
          "取消"
        )
      );
    }
    if (task.status === "failed") {
      rowActions.push(
        el(
          "button",
          {
            class: "btn small ghost",
            type: "button",
            "data-action": "task-retry",
            "data-task": task.task_id,
          },
          iconNode("refresh"),
          "重试"
        )
      );
    }
    body.append(
      el(
        "tr",
        {},
        el("td", {}, KIND_LABELS[task.kind] || task.kind || "-"),
        el("td", {}, statusChip(task.status)),
        el("td", {}, deliveryChip(task)),
        el("td", { class: "mono" }, task.provider || "-"),
        el(
          "td",
          { title: task.prompt || "" },
          (task.prompt || "").slice(0, 48) + ((task.prompt || "").length > 48 ? "…" : "")
        ),
        el("td", { class: "mono" }, formatTime(task.created_at)),
        el("td", { class: "mono" }, duration(task)),
        withActions ? el("td", {}, el("div", { class: "row-actions" }, ...rowActions)) : null
      )
    );
    if (task.error) {
      body.append(
        el(
          "tr",
          {},
          el(
            "td",
            { colspan: withActions ? "8" : "7", class: "mono", style: "color:var(--danger)" },
            task.error
          )
        )
      );
    }
  }
  table.append(body);
  // Narrow screens scroll the table sideways rather than squeezing every
  // column into an unreadable strip.
  return el("div", { class: "table-scroll" }, table);
}

function renderTasks(root) {
  const queue = state.status?.queue || {};
  const stats = el("div", { class: "grid-stats" });
  stats.append(
    statCard("play", "运行中", String(queue.running || 0), `并行上限 ${queue.max_running ?? "-"}`),
    statCard("clock", "等待中", String(queue.waiting || 0), `排队上限 ${queue.max_queued ?? "-"}`),
    statCard("tasks", "最近记录", `<span id="tasks-count">${state.tasks.length}</span>`, "最多显示 100 条")
  );
  const card = el("div", { class: "card" });
  card.append(
    el(
      "div",
      { class: "card-head" },
      el("h2", {}, "任务列表"),
      el("span", { class: "sub" }, "每 5 秒自动刷新"),
      el("span", { class: "spacer" }),
      el(
        "button",
        { class: "btn small ghost", type: "button", "data-action": "tasks-refresh" },
        iconNode("refresh"),
        "刷新"
      )
    )
  );
  const filters = el("div", { class: "tabs task-filters" });
  for (const [value, label] of TASK_FILTERS) {
    filters.append(
      el(
        "button",
        {
          type: "button",
          "data-action": "task-filter",
          "data-state": value,
          "aria-selected": state.taskState === value ? "true" : "false",
        },
        label
      )
    );
  }
  const holder = el("div", { id: "task-holder" }, el("div", { class: "empty" }, "读取中…"));
  card.append(el("div", { class: "task-filter-bar" }, filters), holder);
  root.append(stats, card);
  loadTasks();
  startPolling();
}

async function loadTasks() {
  const holder = $("#task-holder");
  if (!holder) return;
  // Polling rebuilds this table every few seconds. On a phone the sideways
  // position is part of what the reader is looking at, so it is put back.
  const keptScroll = holder.querySelector(".table-scroll")?.scrollLeft || 0;
  const params = new URLSearchParams({ limit: "100" });
  if (state.taskState) params.set("state", state.taskState);
  try {
    const data = await api(`/tasks?${params.toString()}`);
    state.tasks = data.tasks || [];
    const count = $("#tasks-count");
    if (count) count.textContent = String(state.tasks.length);
    holder.replaceChildren();
    if (!state.tasks.length) {
      holder.append(
        el("div", { class: "empty" }, state.taskState ? "这个状态下还没有任务" : "还没有任务")
      );
      return;
    }
    holder.append(taskTable(state.tasks, true));
    const box = holder.querySelector(".table-scroll");
    if (box) box.scrollLeft = keptScroll;
  } catch (error) {
    holder.replaceChildren(el("div", { class: "empty" }, String(error.message || error)));
  }
}

function startPolling() {
  stopPolling();
  if (state.view !== "tasks") return;
  state.polling = window.setInterval(() => {
    loadTasks();
    refreshStatus();
  }, 5000);
}

function stopPolling() {
  if (state.polling) {
    window.clearInterval(state.polling);
    state.polling = 0;
  }
}

async function refreshStatus() {
  try {
    const data = await api("/status");
    state.status = data;
    await loadMetrics();
    renderNav();
    renderStartupBanner();
    if (state.view === "overview") {
      const tasks = await api("/tasks?limit=6").catch(() => null);
      const box = $("#overview-tasks");
      if (tasks && box) {
        box.replaceChildren();
        if (!(tasks.tasks || []).length) {
          box.className = "empty";
          box.textContent = "还没有任务记录";
        } else {
          box.className = "";
          box.append(taskTable(tasks.tasks, false));
        }
        renderNav();
      }
    }
  } catch {
    /* keep the last known status */
  }
}

// -- gallery ----------------------------------------------------------

function renderGallery(root) {
  const card = el("div", { class: "card" });
  card.append(
    el(
      "div",
      { class: "card-head" },
      el("h2", {}, "生成的图片"),
      el("span", { class: "sub", id: "gallery-count" }, gallerySummary()),
      el("span", { class: "spacer" }),
      el(
        "div",
        { class: "tabs" },
        el(
          "button",
          {
            type: "button",
            "data-action": "gallery-tab",
            "data-tab": "all",
            "aria-selected": state.galleryTab === "all" ? "true" : "false",
          },
          "全部"
        ),
        el(
          "button",
          {
            type: "button",
            "data-action": "gallery-tab",
            "data-tab": "fav",
            "aria-selected": state.galleryTab === "fav" ? "true" : "false",
          },
          "收藏夹"
        )
      ),
      el(
        "button",
        { class: "btn small ghost", type: "button", "data-action": "gallery-refresh" },
        iconNode("refresh"),
        "刷新"
      )
    )
  );
  const holder = el("div", { id: "gallery-holder", class: "card-body" }, el("div", { class: "empty" }, "读取中…"));
  card.append(galleryToolbar(), holder);
  root.append(card);
  loadGallery();
}

function galleryToolbar() {
  const search = el("input", {
    class: "gallery-search",
    type: "search",
    id: "gallery-search",
    placeholder: "搜索提示词…",
    value: state.galleryQuery,
  });
  const apply = () => {
    const next = search.value.trim();
    if (next !== state.galleryQuery) state.galleryPicked = [];
    state.galleryQuery = next;
    loadGallery();
  };
  search.addEventListener("input", () => {
    window.clearTimeout(state.galleryTimer);
    state.galleryTimer = window.setTimeout(apply, 300);
  });
  search.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    window.clearTimeout(state.galleryTimer);
    apply();
  });
  const kind = el("select", { class: "gallery-kind", id: "gallery-kind", title: "类型筛选" });
  for (const [value, label] of [
    ["", "全部类型"],
    ["image", "图片"],
    ["selfie", "自拍"],
    ["video", "视频"],
  ]) {
    kind.append(
      el("option", { value, selected: state.galleryKind === value ? "selected" : null }, label)
    );
  }
  kind.addEventListener("change", () => {
    state.galleryKind = kind.value;
    state.galleryPicked = [];
    loadGallery();
  });
  const sort = el("select", { class: "gallery-sort", id: "gallery-sort", title: "排序" });
  for (const [value, label] of [["new", "最新"], ["old", "最早"]]) {
    sort.append(
      el("option", { value, selected: state.gallerySort === value ? "selected" : null }, label)
    );
  }
  sort.addEventListener("change", () => {
    state.gallerySort = sort.value;
    loadGallery();
  });
  return el(
    "div",
    { class: "gallery-toolbar" },
    search,
    kind,
    sort,
    el("span", { class: "spacer" }),
    el(
      "button",
      {
        class: "btn small ghost",
        type: "button",
        "data-action": "gallery-select",
        "aria-pressed": state.gallerySelect ? "true" : "false",
      },
      state.gallerySelect ? "退出多选" : "多选"
    )
  );
}

function gallerySummary() {
  const items = state.gallery || [];
  const favorites = items.filter((item) => item.favorite).length;
  const bytes = items.reduce((total, item) => total + (Number(item.bytes) || 0), 0);
  return `${items.length} 张 · 收藏 ${favorites} · 占用 ${formatBytes(bytes)}`;
}

function galleryKindLabel(kind) {
  const value = String(kind || "image");
  if (value === "video") return "视频";
  if (value === "selfie") return "自拍";
  return "图片";
}

function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value >= 100 || index === 0 ? Math.round(value) : value.toFixed(1)} ${units[index]}`;
}

function visibleGallery() {
  return state.galleryTab === "fav"
    ? state.gallery.filter((item) => item.favorite)
    : state.gallery;
}

function galleryPickedText() {
  return `已选 ${state.galleryPicked.length} 张`;
}

function syncBulkButtons() {
  const empty = state.galleryPicked.length === 0;
  document
    .querySelectorAll("[data-action='bulk-favorite'], [data-action='bulk-delete']")
    .forEach((node) => {
      node.disabled = empty;
    });
}

function galleryBulkBar() {
  const empty = state.galleryPicked.length === 0;
  const inFavorites = state.galleryTab === "fav";
  return el(
    "div",
    { class: "gallery-bulk" },
    el("span", { class: "gallery-picked", id: "gallery-picked" }, galleryPickedText()),
    el("span", { class: "spacer" }),
    el(
      "button",
      { class: "btn small ghost", type: "button", "data-action": "gallery-pick-all" },
      "全选本页"
    ),
    el(
      "button",
      {
        class: "btn small ghost",
        type: "button",
        "data-action": "bulk-favorite",
        disabled: empty,
      },
      iconNode("star"),
      inFavorites ? "取消收藏" : "收藏"
    ),
    el(
      "button",
      {
        class: "btn small ghost danger-text",
        type: "button",
        "data-action": "bulk-delete",
        disabled: empty,
      },
      iconNode("trash"),
      "删除"
    ),
    el(
      "button",
      { class: "btn small ghost", type: "button", "data-action": "bulk-clear" },
      "清空选择"
    )
  );
}

function applyGalleryPick(tile, imageId) {
  if (!imageId) return;
  const index = state.galleryPicked.indexOf(imageId);
  const picked = index < 0;
  if (picked) state.galleryPicked.push(imageId);
  else state.galleryPicked.splice(index, 1);
  if (tile) tile.classList.toggle("picked", picked);
  const box = tile ? tile.querySelector(".tile-check") : null;
  if (box) box.checked = picked;
  const label = $("#gallery-picked");
  if (label) label.textContent = galleryPickedText();
  syncBulkButtons();
}

function selectAllGallery() {
  const shown = visibleGallery();
  const all = shown.length > 0 && shown.every((item) => state.galleryPicked.includes(item.id));
  state.galleryPicked = all ? [] : shown.map((item) => item.id);
  document.querySelectorAll("#gallery-holder .tile").forEach((tile) => {
    const picked = state.galleryPicked.includes(tile.dataset.image);
    tile.classList.toggle("picked", picked);
    const box = tile.querySelector(".tile-check");
    if (box) box.checked = picked;
  });
  const label = $("#gallery-picked");
  if (label) label.textContent = galleryPickedText();
  syncBulkButtons();
}

async function loadGallery(page = 1) {
  const holder = $("#gallery-holder");
  if (!holder) return;
  const wanted = Math.max(1, Math.floor(Number(page) || 1));
  const params = new URLSearchParams({
    offset: String((wanted - 1) * GALLERY_PAGE),
    limit: String(GALLERY_PAGE),
  });
  if (state.galleryQuery) params.set("q", state.galleryQuery);
  if (state.galleryKind) params.set("kind", state.galleryKind);
  if (state.galleryTab === "fav") params.set("favorite", "1");
  params.set("sort", state.gallerySort);
  try {
    const data = await api(`/gallery?${params.toString()}`);
    const items = data.items || [];
    state.galleryTotal = Number(data.total ?? items.length);
    const pages = Math.max(1, Math.ceil(state.galleryTotal / GALLERY_PAGE));
    if (wanted > pages && !items.length) {
      // Deletes shrank the tail: fall back to the new last page.
      return loadGallery(pages);
    }
    state.gallery = items;
    state.galleryPage = Math.min(wanted, pages);
    const shown = visibleGallery();
    const countLabel = $("#gallery-count");
    if (countLabel) countLabel.textContent = gallerySummary();
    const grid = el("div", { class: "gallery-grid" });
    for (const image of shown) grid.append(galleryTile(image));
    if (!shown.length) {
      const filtered = Boolean(state.galleryQuery || state.galleryKind);
      holder.replaceChildren(
        el(
          "div",
          { class: "blank" },
          iconNode("gallery"),
          el(
            "strong",
            {},
            filtered
              ? "没有匹配的结果"
              : state.galleryTab === "fav"
              ? "收藏夹是空的"
              : "画廊还是空的"
          ),
          el(
            "p",
            {},
            filtered
              ? "换个关键词或类型再试试"
              : state.galleryTab === "fav"
              ? "点图片右下角的星标，收进收藏夹"
              : "生成图片或视频之后，结果会自动留档"
          )
        )
      );
      return;
    }
    holder.replaceChildren(grid);
    if (pages > 1) holder.append(galleryPager(pages));
    if (state.gallerySelect) holder.append(galleryBulkBar());
  } catch (error) {
    holder.replaceChildren(el("div", { class: "empty" }, String(error.message || error)));
  }
}

function galleryPager(pages) {
  const page = state.galleryPage;
  const jump = el("input", {
    class: "pager-jump",
    type: "number",
    min: "1",
    max: String(pages),
    step: "1",
    value: String(page),
    "aria-label": "当前页码，输入后回车跳转",
  });
  const commit = () => gotoGalleryPage(jump.value, pages);
  jump.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    commit();
  });
  jump.addEventListener("change", commit);
  const nav = (target, label, glyph, disabled) =>
    el(
      "button",
      {
        class: "btn small ghost pager-nav",
        type: "button",
        "data-action": "gallery-page",
        "data-page": String(target),
        disabled: disabled ? "disabled" : null,
      },
      iconNode(glyph),
      label
    );
  return el(
    "div",
    { class: "gallery-pager" },
    nav(page - 1, "上一页", "chevron-left", page <= 1),
    el("span", { class: "pager-jump-box" }, jump, el("span", { class: "pager-total" }, `/ ${pages}`)),
    nav(page + 1, "下一页", "chevron-right", page >= pages)
  );
}

function gotoGalleryPage(value, pages) {
  const wanted = Math.floor(Number(value));
  if (!Number.isFinite(wanted)) return;
  const target = Math.min(Math.max(wanted, 1), Math.max(1, pages));
  if (target === state.galleryPage) return;
  loadGallery(target);
}

function galleryTile(image) {
  const tile = el("div", { class: "tile" });
  tile.dataset.image = image.id;
  const isVideo = String(image.kind || "image") === "video";
  if (isVideo) {
    tile.append(el("div", { class: "shot video-placeholder" }, iconNode("play")));
  } else {
    const shot = el("img", { class: "shot", alt: "", loading: "lazy" });
    shot.dataset.imageId = image.id;
    loadThumb(shot, image.id);
    tile.append(shot);
  }
  if (state.gallerySelect) {
    const picked = state.galleryPicked.includes(image.id);
    tile.classList.toggle("picked", picked);
    tile.append(
      el("input", {
        class: "tile-check",
        type: "checkbox",
        "aria-label": "选择图片",
        checked: picked ? "checked" : null,
        "data-action": "gallery-pick",
        "data-image": image.id,
      })
    );
  }
  tile.append(
    el(
      "div",
      { class: "tile-foot" },
      el("span", {}, galleryKindLabel(image.kind)),
      el("span", {}, image.size || ""),
      el("span", { class: "spacer" }),
      el(
        "button",
        {
          class: "btn ghost icon-btn small",
          type: "button",
          title: "删除",
          "data-action": "gallery-delete",
          "data-image": image.id,
        },
        iconNode("trash")
      )
    )
  );
  tile.append(
    el(
      "button",
      {
        class: "fav",
        type: "button",
        title: image.favorite ? "取消收藏" : "收藏",
        "data-on": image.favorite ? "true" : "false",
        "data-action": "gallery-favorite",
        "data-image": image.id,
        "data-favorite": image.favorite ? "1" : "0",
      },
      iconNode("star")
    )
  );
  return tile;
}

function loadThumb(img, imageId) {
  img.src = `${BASE}/gallery/thumb?image_id=${encodeURIComponent(imageId)}`;
  img.onerror = () => {
    img.alt = "图片读取失败";
  };
}

async function fetchModels() {
  const provider = selectedProvider();
  if (!provider) return;
  const raw = provider.api_keys;
  const keys = Array.isArray(raw) ? raw : String(raw || "").split("\n");
  toast("正在获取模型列表…");
  try {
    const data = await api("/models", {
      method: "POST",
      body: JSON.stringify({
        base_url: provider.base_url || "",
        server_url: provider.server_url || "",
        api_key: keys.map((item) => String(item).trim()).filter(Boolean)[0] || "",
      }),
    });
    const models = data.models || [];
    if (!models.length) {
      toast("接口没有返回模型，检查地址和密钥", "error");
      return;
    }
    if (models.length === 1) {
      provider.model = models[0];
      closeImageModal();
      render();
      toast("已填入唯一的模型");
      return;
    }
    showModelPicker(models, provider.model || "");
  } catch (error) {
    toast(String(error.message || error), "error");
  }
}

function showModelPicker(models, selected) {
  const modal = $("#modal");
  modal.replaceChildren();
  const card = el("div", { class: "modal-card picker-card" });
  card.append(
    el(
      "button",
      { class: "icon-btn picker-close", type: "button", "data-action": "modal-close", title: "关闭" },
      iconNode("close")
    )
  );
  const list = el("div", { class: "model-list" });
  let picked = state.pickerModel || selected || "";
  const buttons = [];
  const mark = () => {
    for (const button of buttons) {
      const active = button.dataset.model === picked && picked !== "";
      button.classList.toggle("picked", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    }
  };
  for (const name of models) {
    const button = el(
      "button",
      { class: "model-item", type: "button", "data-model": name, "aria-pressed": "false" },
      el("span", { class: "model-name" }, name),
      el("span", { class: "model-check" }, iconNode("check"))
    );
    button.addEventListener("click", () => {
      const provider = (state.config.providers || []).find((item) => item.id === state.selected);
      if (provider) provider.model = name;
      state.pickerModel = name;
      picked = name;
      mark();
      render();
    });
    buttons.push(button);
    list.append(button);
  }
  card.append(list);
  modal.append(card);
  modal.classList.remove("hidden");
  mark();
}

function closeImageModal() {
  const modal = $("#modal");
  modal.classList.add("hidden");
}

function openImage(imageId) {
  const image = state.gallery.find((item) => item.id === imageId);
  const modal = $("#modal");
  modal.replaceChildren();
  const favorited = Boolean(image?.favorite);
  const shell = el("div", { class: "modal-shell" });
  shell.append(
    el(
      "div",
      { class: "modal-head" },
      el("h3", { class: "modal-title" }, "图片"),
      el("span", { class: "spacer" }),
      el(
        "button",
        { class: "icon-btn", type: "button", "data-action": "modal-close", title: "关闭" },
        iconNode("close")
      )
    )
  );

  const isVideo = String(image?.kind || "image") === "video";
  const picture = isVideo
    ? el("video", { controls: "controls", playsinline: "" })
    : el("img", { alt: "" });
  const canvas = el("div", { class: "modal-canvas" }, picture);

  const info = el("div", { class: "modal-info" });
  info.append(
    el(
      "div",
      { class: "modal-meta" },
      [image?.model, image?.size].filter(Boolean).join(" · ") || "生成结果"
    )
  );
  const facts = el("div", { class: "modal-facts", id: "image-facts" });
  facts.dataset.image = imageId;
  info.append(facts);
  info.append(el("h4", { class: "modal-section-title" }, "生成提示词"));
  info.append(
    el(
      "pre",
      { class: "modal-prompt-text" },
      String(image?.prompt || "").trim() || "这条记录没有留下提示词"
    )
  );
  info.append(
    el(
      "div",
      { class: "modal-sub" },
      [image?.provider, formatStamp(image?.created_at)].filter(Boolean).join(" · ")
    )
  );
  const actions = el("div", { class: "modal-actions" });
  actions.append(
    el(
      "button",
      {
        class: favorited ? "btn small ghost is-on" : "btn small ghost",
        type: "button",
        "data-action": "image-favorite",
        "data-image": imageId,
        "data-favorite": favorited ? "1" : "0",
      },
      iconNode("star"),
      favorited ? "取消收藏" : "收藏"
    ),
    el(
      "button",
      { class: "btn small ghost danger-text", type: "button", "data-action": "image-delete", "data-image": imageId },
      iconNode("trash"),
      "删除"
    ),
    el(
      "button",
      { class: "btn small", type: "button", "data-action": "image-download", "data-image": imageId },
      iconNode("save"),
      "下载原图"
    )
  );
  info.append(actions);

  shell.append(el("div", { class: "modal-main" }, canvas, info));
  modal.append(shell);
  modal.classList.remove("hidden");
  const viewId = encodeURIComponent(imageId);
  // The modal never pulls the original file: preview is a 2048px JPEG, with
  // the grid thumb and only then the original as fallbacks.
  picture.src = `${BASE}/gallery/preview?image_id=${viewId}`;
  if (!isVideo) {
    picture.onerror = () => {
      const step = picture.dataset.fallback || "";
      if (!step) {
        picture.dataset.fallback = "thumb";
        picture.src = `${BASE}/gallery/thumb?image_id=${viewId}`;
      } else if (step === "thumb") {
        picture.dataset.fallback = "file";
        picture.src = `${BASE}/gallery/file?image_id=${viewId}`;
      }
    };
  }
  loadImageDetail(imageId);
}

function imageDetailNodes(item) {
  const rows = [
    ["服务商", item.provider],
    ["模型", item.model],
    ["尺寸", item.size],
    ["类型", galleryKindLabel(item.kind)],
    ["创建时间", formatStamp(item.created_at)],
    ["收藏", item.favorite ? "是" : "否"],
    ["文件大小", formatBytes(Number(item.bytes) || 0)],
    ["文件名", item.filename],
  ];
  return rows
    .filter(([, value]) => String(value ?? "").trim() !== "")
    .map(([label, value]) =>
      el(
        "div",
        { class: "modal-fact" },
        el("span", { class: "k" }, label),
        el("span", { class: "v" }, String(value))
      )
    );
}

async function loadImageDetail(imageId) {
  const box = $("#image-facts");
  if (!box || !imageId) return;
  try {
    const data = await api(`/gallery/item/${encodeURIComponent(imageId)}`);
    if (box.dataset.image !== imageId) return;
    if (!data.ok || !data.item) throw new Error(data.error || "详情读取失败");
    box.replaceChildren(...imageDetailNodes(data.item));
  } catch (error) {
    if (box.dataset.image !== imageId) return;
    const cached = (state.gallery || []).find((row) => row.id === imageId);
    box.replaceChildren(
      ...(cached
        ? imageDetailNodes(cached)
        : [el("div", { class: "modal-fact" }, "详情读取失败")])
    );
  }
}

function formatStamp(value) {
  const stamp = Number(value || 0);
  if (!stamp) return "";
  const date = new Date(stamp < 1e12 ? stamp * 1000 : stamp);
  if (Number.isNaN(date.getTime())) return "";
  return `${date.getMonth() + 1} 月 ${date.getDate()} 日 ${String(date.getHours()).padStart(2, "0")}:${String(
    date.getMinutes()
  ).padStart(2, "0")}`;
}

function downloadImage(imageId) {
  const image = state.gallery.find((item) => item.id === imageId);
  const link = document.createElement("a");
  link.href = `${BASE}/gallery/file?image_id=${encodeURIComponent(imageId)}`;
  link.download = image?.filename || `aiimg-${imageId}`;
  document.body.append(link);
  link.click();
  link.remove();
}

// -- settings ---------------------------------------------------------

function referenceFilesField(item, value, companion = null) {
  const files = state.refFiles;
  const wrap = el("div", { class: "field wide" });
  const block = el("div", { class: "ref-block" });
  block.append(
    el(
      "div",
      { class: "field-head" },
      el("label", {}, item.name || "自拍参考图文件")
    )
  );
  if (item.hint) block.append(el("div", { class: "hint" }, item.hint));
  block.append(
    el(
      "div",
      { class: "ref-actions" },
      el(
        "button",
        { class: "btn small ref-manage", type: "button", "data-action": "refs-manage" },
        iconNode("image"),
        "管理文件"
      ),
      el("span", { class: "file-count", id: "ref-count" }, `文件：${files.length}`)
    )
  );
  const row = el("div", { class: "field-row ref-row" }, block);
  if (companion) row.append(companion);
  wrap.append(row);
  return wrap;
}

// 开关和它的列表并排：左边开关，右边列表摘要，两列各自贴顶。
function switchListRow(switchField, listSummary) {
  const row = el("div", { class: "field-row feature-row" }, switchField, listSummary);
  return el("div", { class: "field wide" }, row);
}

function toLines(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => String(entry ?? "").trim()).filter(Boolean);
  }
  return String(value ?? "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function parsePresets(value) {
  const items = [];
  for (const raw of toLines(value)) {
    const text = String(raw || "").trim();
    if (!text) continue;
    const index = text.indexOf(":");
    if (index <= 0) continue;
    const name = text.slice(0, index).trim();
    const prompt = text.slice(index + 1).trim();
    if (!name || !prompt) continue;
    items.push({ name, prompt, raw: text });
  }
  return items;
}

function shortText(text, limit = 20) {
  const value = String(text || "");
  return value.length > limit ? `${value.slice(0, limit)}…` : value;
}

function presetField(sectionKey, field, item, value) {
  const presets = parsePresets(value);
  const wrap = el("div", { class: "field wide" });
  wrap.append(
    el(
      "div",
      { class: "field-head" },
      el("label", {}, item.name || "预设提示词")
    )
  );
  const row = el("div", { class: "preset-row" });
  if (!presets.length) {
    row.append(el("span", { class: "hint" }, "还没有预设"));
  } else {
    row.append(
      el(
        "span",
        { class: "chip", title: presets[0].prompt },
        `${presets[0].name}:${shortText(presets[0].prompt, 16)}`
      )
    );
    if (presets.length > 1) {
      row.append(el("span", { class: "chip chip-more" }, `+${presets.length - 1}`));
    }
  }
  row.append(
    el(
      "button",
      {
        class: "btn small ghost chip-add",
        type: "button",
        "data-action": "presets-edit",
        "data-section": sectionKey,
        "data-section-field": field,
      },
      iconNode(presets.length ? "edit" : "plus"),
      presets.length ? "编辑" : "添加"
    )
  );
  wrap.append(row);
  if (item.hint) wrap.append(el("div", { class: "hint" }, item.hint));
  return wrap;
}

const REFERENCE_ROLE_LABELS = {
  subject: "主体",
  clothing: "服装",
  style: "画风",
  pose: "姿势",
  background: "背景",
};

function openReferenceManager() {
  const modal = $("#modal");
  modal.replaceChildren();
  const files = state.refFiles;
  const card = el("div", { class: "modal-shell narrow" });
  card.append(
    el(
      "div",
      { class: "modal-head" },
      el("h3", { class: "modal-title" }, "自拍参考图"),
      el("span", { class: "spacer" }),
      el(
        "button",
        { class: "icon-btn", type: "button", "data-action": "modal-close", title: "关闭" },
        iconNode("close")
      )
    )
  );
  const body = el("div", { class: "modal-body" });
  const grid = el("div", { class: "ref-grid" });
  if (!files.length) {
    grid.append(el("div", { class: "blank small" }, el("p", {}, "还没有上传，点下面的按钮加一张")));
  }
  files.forEach((file, index) => {
    const name = file.name;
    const tile = el("div", { class: "ref-tile" });
    const thumb = el("div", { class: "ref-thumb" });
    const img = el("img", { alt: "" });
    api(`/files/raw?name=${encodeURIComponent(name)}`)
      .then((data) => {
        if (data?.data_url) img.src = data.data_url;
      })
      .catch(() => {});
    thumb.append(
      img,
      el(
        "button",
        { class: "icon-btn danger", type: "button", "data-action": "refs-delete", "data-name": name, title: "删除" },
        iconNode("trash")
      )
    );
    const select = el(
      "select",
      { class: "ref-role", "data-name": name, title: "参考图角色" },
      ...Object.entries(REFERENCE_ROLE_LABELS).map(([role, label]) =>
        el("option", { value: role }, label)
      )
    );
    select.value = REFERENCE_ROLE_LABELS[file.role] ? file.role : "subject";
    select.addEventListener("change", () => setReferenceRole(name, select.value));
    const move = el(
      "div",
      { class: "ref-move" },
      el(
        "button",
        {
          class: "icon-btn",
          type: "button",
          "data-action": "refs-up",
          "data-name": name,
          title: "上移",
          disabled: index === 0,
        },
        iconNode("up")
      ),
      el(
        "button",
        {
          class: "icon-btn",
          type: "button",
          "data-action": "refs-down",
          "data-name": name,
          title: "下移",
          disabled: index === files.length - 1,
        },
        iconNode("down")
      )
    );
    tile.append(thumb, el("div", { class: "ref-controls" }, select, move));
    grid.append(tile);
  });
  body.append(grid);
  const input = el("input", { type: "file", accept: "image/*", multiple: true, class: "hidden" });
  input.addEventListener("change", () => uploadReferences(input.files));
  body.append(
    el(
      "div",
      { class: "modal-actions" },
      el(
        "button",
        { class: "btn small", type: "button", "data-action": "refs-upload" },
        iconNode("plus"),
        "上传图片"
      ),
      input,
      el("span", { class: "spacer" }),
      el("span", { class: "hint" }, "支持 jpg/png/webp，单张不超过 10MB")
    )
  );
  card.append(body);
  modal.append(card);
  modal.classList.remove("hidden");
  modal.querySelector("[data-action='refs-upload']").addEventListener("click", () => input.click());
  modal.querySelectorAll("[data-action='refs-delete']").forEach((button) => {
    button.addEventListener("click", () => deleteReference(button.dataset.name));
  });
  modal.querySelectorAll("[data-action='refs-up']").forEach((button) => {
    button.addEventListener("click", () => moveReference(button.dataset.name, -1));
  });
  modal.querySelectorAll("[data-action='refs-down']").forEach((button) => {
    button.addEventListener("click", () => moveReference(button.dataset.name, 1));
  });
}

async function uploadReferences(fileList) {
  const files = Array.from(fileList || []);
  let changed = false;
  for (const file of files) {
    if (file.size > 10 * 1024 * 1024) {
      toast(`${file.name} 超过 10MB，跳过`, "error");
      continue;
    }
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error("读取文件失败"));
      reader.readAsDataURL(file);
    });
    try {
      const data = await api("/files/upload", {
        method: "POST",
        body: JSON.stringify({ name: file.name, data_url: dataUrl }),
      });
      if (data?.files) {
        applyReferenceFiles(data.files);
        changed = true;
      }
    } catch (error) {
      toast(String(error.message || error), "error");
    }
  }
  if (changed) toast("参考图已更新");
}

// The manager is a modal on top of the page, so both the field behind it and
// the open grid have to be redrawn by hand after an upload or a delete. The
// list never touches the config, which keeps the revision stable for saving.
function applyReferenceFiles(files) {
  state.refFiles = normalizeReferenceFiles(files);
  const modal = $("#modal");
  const managerOpen = !modal.classList.contains("hidden") && !!modal.querySelector(".ref-grid");
  render();
  if (managerOpen) openReferenceManager();
}

async function deleteReference(name) {
  try {
    const data = await api("/files/delete", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    applyReferenceFiles(data?.files || []);
  } catch (error) {
    toast(String(error.message || error), "error");
  }
}

async function setReferenceRole(name, role) {
  try {
    const data = await api("/files/role", {
      method: "POST",
      body: JSON.stringify({ name, role }),
    });
    if (data?.files) applyReferenceFiles(data.files);
  } catch (error) {
    toast(String(error.message || error), "error");
    openReferenceManager();
  }
}

// Reordering posts the full name list: the backend rejects a partial one.
async function moveReference(name, delta) {
  const names = state.refFiles.map((item) => item.name);
  const index = names.indexOf(name);
  const target = index + delta;
  if (index < 0 || target < 0 || target >= names.length) return;
  [names[index], names[target]] = [names[target], names[index]];
  try {
    const data = await api("/files/order", {
      method: "POST",
      body: JSON.stringify({ names }),
    });
    if (data?.files) applyReferenceFiles(data.files);
  } catch (error) {
    toast(String(error.message || error), "error");
  }
}

function listValueItems(value) {
  return toLines(value)
    .map((line) => String(line || "").trim())
    .filter(Boolean);
}

// Trim, drop blanks and collapse duplicates while keeping the typed order.
function normalizeListItems(items) {
  const seen = new Set();
  const out = [];
  for (const raw of items || []) {
    const text = String(raw ?? "").trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

function listSummaryField(options) {
  const wrap = el("div", { class: "field wide" });
  wrap.append(
    el(
      "div",
      { class: "field-head" },
      el("label", {}, options.label || options.title),
      el(
        "button",
        {
          class: "btn small ghost",
          type: "button",
          "data-action": "list-edit",
          "data-list-target": options.target,
          "data-list-title": options.title,
        },
        iconNode(options.items.length ? "edit" : "plus"),
        options.items.length ? `编辑（${options.items.length}）` : "添加"
      )
    )
  );
  const row = el("div", { class: "preset-row" });
  if (!options.items.length) {
    row.append(el("span", { class: "hint" }, "还没有一项"));
  } else {
    options.items.slice(0, 3).forEach((text) => {
      row.append(el("span", { class: "chip", title: text }, shortText(text, 18)));
    });
    if (options.items.length > 3) {
      row.append(el("span", { class: "chip chip-more" }, `+${options.items.length - 3}`));
    }
  }
  wrap.append(row);
  if (options.hint) wrap.append(el("div", { class: "hint" }, options.hint));
  return wrap;
}

function listField(sectionKey, field, item, value) {
  const label = item.name || fieldLabel(field);
  return listSummaryField({
    label,
    title: label.endsWith("列表") ? label : `${label}列表`,
    hint: item.hint || "",
    items: listValueItems(value),
    target: `section:${sectionKey}:${field}`,
  });
}

function providerKeysField(provider) {
  const items = (Array.isArray(provider.api_keys) ? provider.api_keys : [])
    .map((key) => String(key || "").trim())
    .filter(Boolean);
  return listSummaryField({
    label: fieldLabel("api_keys"),
    title: `${fieldLabel("api_keys")}列表`,
    hint: "多把密钥轮流使用，被服务商拒绝的那把会暂时停用。",
    items,
    target: "provider:api_keys",
  });
}

function openListEditorFor(target, title) {
  const parts = String(target || "").split(":");
  if (parts[0] === "section") {
    const [, sectionKey, field] = parts;
    if (!sectionKey || !field) return;
    state.config[sectionKey] = state.config[sectionKey] || {};
    openListEditor({
      title,
      items: listValueItems(state.config[sectionKey][field]),
      onConfirm: (draft) => {
        state.config[sectionKey][field] = draft.join("\n");
        render();
        updateTopbar();
      },
    });
    return;
  }
  if (parts[0] === "provider") {
    const provider = selectedProvider();
    const field = parts[1];
    if (!provider || !field) return;
    openListEditor({
      title,
      items: (Array.isArray(provider[field]) ? provider[field] : listValueItems(provider[field])).map((item) =>
        String(item || "").trim()
      ),
      onConfirm: (draft) => {
        provider[field] = draft;
        render();
        updateTopbar();
      },
    });
  }
}

function openListEditor(options) {
  const modal = $("#modal");
  modal.replaceChildren();
  const draft = normalizeListItems(options.items);
  // Keep the list as it was before the editor opened, even when a paste reopens it.
  const original = normalizeListItems(options.original ?? options.items);
  const card = el("div", { class: "modal-shell narrow" });
  const countChip = el("span", { class: "chip count" }, `${draft.length} 项`);
  card.append(
    el(
      "div",
      { class: "modal-head" },
      el("h3", { class: "modal-title" }, options.title || "列表"),
      countChip,
      el("span", { class: "spacer" }),
      el(
        "button",
        { class: "icon-btn", type: "button", "data-action": "modal-close", title: "关闭" },
        iconNode("close")
      )
    )
  );
  const body = el("div", { class: "modal-body" });
  const addInput = el("input", {
    type: "text",
    class: "compose-input",
    placeholder: "输入一项后按回车添加",
    spellcheck: "false",
  });
  const addButton = el(
    "button",
    { class: "compose-send", type: "button", title: "添加到列表" },
    iconNode("plus")
  );
  addButton.disabled = true;
  body.append(el("div", { class: "compose" }, addInput, addButton));

  const grid = el("div", { class: "entry-grid" });
  body.append(grid);

  const paint = (scrollToEnd = false) => {
    countChip.textContent = `${draft.length} 项`;
    grid.replaceChildren();
    if (!draft.length) {
      grid.append(
        el(
          "div",
          { class: "entry-hollow" },
          el("span", { class: "entry-hollow-title" }, "还没有任何一项"),
          el("span", { class: "entry-hollow-tip" }, "敲回车添加，或者用下面的批量粘贴一次贴进来")
        )
      );
      return;
    }
    draft.forEach((text, index) => {
      const row = el("div", { class: "entry" });
      row.append(el("span", { class: "entry-no" }, String(index + 1).padStart(2, "0")));
      row.append(el("span", { class: "entry-text", title: text }, text));
      const remove = el(
        "button",
        { class: "entry-drop", type: "button", title: "删除这一项" },
        iconNode("trash")
      );
      remove.addEventListener("click", () => {
        draft.splice(index, 1);
        paint();
      });
      row.append(remove);
      grid.append(row);
    });
    // A fresh row lands at the bottom; bring it into view.
    if (scrollToEnd) grid.scrollTop = grid.scrollHeight;
  };

  const addCurrent = () => {
    const text = String(addInput.value || "").trim();
    if (!text) return;
    if (!draft.includes(text)) draft.push(text);
    addInput.value = "";
    addButton.disabled = true;
    paint(true);
    addInput.focus();
  };
  addInput.addEventListener("input", () => {
    addButton.disabled = !String(addInput.value || "").trim();
  });
  addInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    addCurrent();
  });
  addButton.addEventListener("click", addCurrent);

  const reopen = (lines) => {
    const next = normalizeListItems([...draft, ...(lines || [])]);
    openListEditor({
      ...options,
      items: next,
      original,
      scrollToEnd: Boolean(lines && lines.length),
    });
  };
  const importButton = el(
    "button",
    { class: "btn small ghost", type: "button", "data-action": "list-import" },
    iconNode("copy"),
    "批量粘贴"
  );
  importButton.addEventListener("click", () =>
    openListImport({ onImport: reopen, onCancel: () => reopen() })
  );

  body.append(
    el(
      "div",
      { class: "modal-actions" },
      importButton,
      el("span", { class: "spacer" }),
      el("button", { class: "btn small ghost", type: "button", "data-action": "modal-close" }, "取消"),
      el(
        "button",
        { class: "btn small", type: "button", "data-action": "list-save" },
        iconNode("check"),
        "保存"
      )
    )
  );
  card.append(body);
  modal.append(card);
  modal.classList.remove("hidden");
  // An untouched list still closes cleanly, it just has nothing to write back.
  const save = () => {
    const clean = normalizeListItems(draft);
    const untouched =
      clean.length === original.length && clean.every((text, index) => text === original[index]);
    closeModal();
    if (!untouched) options.onConfirm(clean);
  };
  card.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    event.stopPropagation();
    closeModal();
  });
  card.querySelector("[data-action='list-save']").addEventListener("click", save);
  paint(Boolean(options.scrollToEnd));
  addInput.focus();
}

function openListImport(options) {
  const modal = $("#modal");
  modal.replaceChildren();
  const card = el("div", { class: "modal-shell narrow" });
  const closeButton = el(
    "button",
    { class: "icon-btn", type: "button", title: "关闭" },
    iconNode("close")
  );
  closeButton.addEventListener("click", () => options.onCancel());
  card.append(
    el(
      "div",
      { class: "modal-head" },
      el("h3", { class: "modal-title" }, "批量粘贴"),
      el("span", { class: "spacer" }),
      closeButton
    )
  );
  const body = el("div", { class: "modal-body" });
  const area = el("textarea", {
    rows: "10",
    class: "paste-area",
    placeholder: "第一项\n第二项\n第三项",
    spellcheck: "false",
  });
  const countLabel = el("span", { class: "paste-count" }, "0 项");
  body.append(
    el(
      "div",
      { class: "paste-shell" },
      el(
        "div",
        { class: "paste-head" },
        el("span", { class: "paste-label" }, "每行一项"),
        countLabel
      ),
      area
    )
  );
  body.append(
    el(
      "p",
      { class: "hint paste-tip" },
      "空行会自动跳过，重复的也不会重复添加。Ctrl / ⌘ + Enter 直接导入。"
    )
  );
  const importButton = el(
    "button",
    { class: "btn small", type: "button", "data-action": "list-import-save" },
    iconNode("check"),
    "导入"
  );
  importButton.disabled = true;
  const cancelButton = el("button", { class: "btn small ghost", type: "button" }, "取消");
  cancelButton.addEventListener("click", () => options.onCancel());
  body.append(
    el("div", { class: "modal-actions" }, el("span", { class: "spacer" }), cancelButton, importButton)
  );
  const readLines = () =>
    String(area.value || "")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
  const commit = () => {
    const lines = readLines();
    if (!lines.length) return;
    options.onImport(lines);
  };
  area.addEventListener("input", () => {
    const count = readLines().length;
    countLabel.textContent = `${count} 项`;
    importButton.disabled = count === 0;
  });
  importButton.addEventListener("click", commit);
  card.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      options.onCancel();
      return;
    }
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      commit();
    }
  });
  card.append(body);
  modal.append(card);
  modal.classList.remove("hidden");
  area.focus();
}

function openPresetEditor(sectionKey, field) {
  const modal = $("#modal");
  modal.replaceChildren();
  const list = parsePresets(state.config[sectionKey]?.[field] || []);
  const card = el("div", { class: "modal-shell narrow" });
  card.append(
    el(
      "div",
      { class: "modal-head" },
      el("h3", { class: "modal-title" }, "预设提示词"),
      el("span", { class: "spacer" }),
      el(
        "button",
        { class: "icon-btn", type: "button", "data-action": "modal-close", title: "关闭" },
        iconNode("close")
      )
    )
  );
  const body = el("div", { class: "modal-body" });
  const rows = el("div", { class: "preset-rows" });
  const addRow = (name = "", prompt = "") => {
    const row = el("div", { class: "preset-edit-row" });
    row.append(
      el("input", { type: "text", class: "preset-name", value: name, placeholder: "预设名" }),
      el("input", { type: "text", class: "preset-prompt", value: prompt, placeholder: "英文提示词" }),
      el(
        "button",
        { class: "icon-btn danger", type: "button", title: "删除" },
        iconNode("trash")
      )
    );
    row.querySelector("button").addEventListener("click", () => row.remove());
    rows.append(row);
  };
  const addBulkRow = () => {
    const wrap = el("div", { class: "preset-bulk-row" });
    const area = el("textarea", {
      rows: "3",
      class: "preset-bulk",
      placeholder: "每行一个，格式：预设名:英文提示词",
      spellcheck: "false",
    });
    const importButton = el(
      "button",
      { class: "btn small", type: "button", "data-action": "presets-import" },
      iconNode("check"),
      "导入"
    );
    importButton.addEventListener("click", () => {
      const lines = area.value.split(/\r?\n/);
      wrap.remove();
      let added = 0;
      for (const line of lines) {
        const text = String(line || "").trim();
        const index = text.indexOf(":");
        if (index <= 0) continue;
        const name = text.slice(0, index).trim();
        const prompt = text.slice(index + 1).trim();
        if (!name || !prompt) continue;
        addRow(name, prompt);
        added += 1;
      }
      if (!added) toast("没有读到可用的行，格式是 预设名:提示词", "error");
    });
    wrap.append(area, importButton);
    rows.append(wrap);
    area.focus();
  };
  list.forEach((item) => addRow(item.name, item.prompt));
  if (!list.length) addRow();
  body.append(rows);
  body.append(
    el(
      "div",
      { class: "modal-actions" },
      el(
        "button",
        { class: "btn small ghost", type: "button", "data-action": "presets-add-row" },
        iconNode("plus"),
        "添加一行"
      ),
      el(
        "button",
        { class: "btn small ghost", type: "button", "data-action": "presets-bulk" },
        iconNode("copy"),
        "批量添加"
      ),
      el("span", { class: "spacer" }),
      el(
        "button",
        { class: "btn small", type: "button", "data-action": "presets-save" },
        iconNode("check"),
        "保存"
      )
    )
  );
  card.append(body);
  modal.append(card);
  modal.classList.remove("hidden");
  modal.querySelector("[data-action='presets-add-row']").addEventListener("click", () => addRow());
  modal.querySelector("[data-action='presets-bulk']").addEventListener("click", () => addBulkRow());
  modal.querySelector("[data-action='presets-save']").addEventListener("click", () => {
    const lines = [];
    modal.querySelectorAll(".preset-edit-row").forEach((row) => {
      const name = row.querySelector(".preset-name").value.trim();
      const prompt = row.querySelector(".preset-prompt").value.trim();
      if (name && prompt) lines.push(`${name}:${prompt}`);
    });
    state.config[sectionKey][field] = lines.join("\n");
    closeModal();
    render();
  });
}

// Textareas grow with their content instead of trapping it in a fixed box.
const TEXTAREA_MIN = 74;
const TEXTAREA_MAX = 320;

function fitTextarea(area) {
  if (!area || !area.isConnected) return;
  area.style.height = "auto";
  const wanted = area.scrollHeight + 2;
  area.style.height = `${Math.min(Math.max(wanted, TEXTAREA_MIN), TEXTAREA_MAX)}px`;
  area.style.overflowY = wanted > TEXTAREA_MAX ? "auto" : "hidden";
}

function fitTextareas(root) {
  for (const area of (root || document).querySelectorAll("textarea[data-autosize]")) {
    fitTextarea(area);
  }
}

function sectionInput(sectionKey, field, item, value) {
  const kind = item.type || "string";
  if (sectionKey === "section_features" && field === "selfie_reference_files") {
    return referenceFilesField(item, value);
  }
  if ((sectionKey === "section_features" || sectionKey === "section_prompts") &&
      (field === "draw_presets" || field === "edit_presets")) {
    return presetField(sectionKey, field, item, value);
  }
  // 视频白名单和别处的白名单共用同一个列表编辑器。
  if (sectionKey === "section_features" && field === "video_whitelist") {
    return listField(sectionKey, field, item, value);
  }
  if (sectionKey === "section_access" && field === "whitelist") {
    return listField(sectionKey, field, item, value);
  }
  if (sectionKey === "section_advanced" && field === "trusted_result_origins") {
    return listField(sectionKey, field, item, value);
  }
  const wide = ["textarea", "list", "json", "editor", "markdown"].includes(kind);
  const compact = kind === "integer" || kind === "float" || kind === "select";
  const wrap = el("div", { class: `field${wide ? " wide" : ""}${compact ? " compact" : ""}` });
  wrap.append(el("label", {}, item.name || field));
  let control;
  if (kind === "switch") {
    control = el(
      "label",
      { class: "switch" },
      el("input", {
        type: "checkbox",
        checked: value === true,
        "data-section": sectionKey,
        "data-section-field": field,
        "data-kind": "bool",
      }),
      el("span", { class: "track" }),
      el("span", { class: "switch-text" }, value === true ? "已开启" : "已关闭")
    );
  } else if (kind === "select") {
    const choices = Array.isArray(item.options) ? item.options.map(String) : [];
    const current = value === undefined || value === null ? "" : String(value);
    control = el(
      "select",
      {
        "data-section": sectionKey,
        "data-section-field": field,
        "data-kind": "text",
      },
      ...(current && !choices.includes(current)
        ? [el("option", { value: current }, current)]
        : []),
      ...choices.map((option) =>
        el("option", { value: option, selected: option === current ? true : false }, option)
      )
    );
  } else if (kind === "integer" || kind === "float") {
    control = el("input", {
      type: "number",
      value: value ?? "",
      step: kind === "float" ? "any" : "1",
      min: item.minimum ?? item.min ?? "",
      max: item.maximum ?? item.max ?? "",
      "data-section": sectionKey,
      "data-section-field": field,
      "data-kind": kind === "float" ? "float" : "int",
    });
  } else if (kind === "list") {
    control = el("textarea", {
      class: "grow",
      "data-autosize": "1",
      "data-section": sectionKey,
      "data-section-field": field,
      "data-kind": "list",
      spellcheck: "false",
    });
    control.value = (Array.isArray(value) ? value : []).join("\n");
    control.addEventListener("input", () => fitTextarea(control));
  } else if (kind === "textarea" || kind === "json" || kind === "editor" || kind === "markdown") {
    control = el("textarea", {
      class: "grow",
      "data-autosize": "1",
      "data-section": sectionKey,
      "data-section-field": field,
      "data-kind": kind === "json" ? "json" : "text",
      spellcheck: "false",
    });
    control.value = kind === "json" ? JSON.stringify(value ?? {}, null, 2) : value ?? "";
    control.addEventListener("input", () => fitTextarea(control));
  } else {
    control = el("input", {
      type: "text",
      value: value ?? "",
      "data-section": sectionKey,
      "data-section-field": field,
      "data-kind": "text",
      spellcheck: "false",
    });
  }
  wrap.append(control);
  if (item.hint) wrap.append(el("div", { class: "hint" }, item.hint));
  return wrap;
}

function compatCoverage() {
  const providers = state.config.providers || [];
  const hasImage = providers.some((item) => item.template_key !== "agnes_video");
  const hasVideo = providers.some((item) => item.template_key === "agnes_video");
  const selfieFiles = state.refFiles;
  const selfieReady =
    Boolean(selfieFiles.length) || state.config.section_features?.selfie_use_persona !== false;
  const parts = [];
  if (hasImage) parts.push("<img>");
  if (selfieReady) parts.push("<selfie>");
  if (hasVideo) parts.push("<video>");
  return parts;
}

function renderSettings(root) {
  for (const [sectionKey, definition] of Object.entries(state.sections)) {
    const values = state.config[sectionKey] || {};
    const card = el("div", { class: "card" });
    card.append(
      el(
        "div",
        { class: "card-head" },
        el("h2", {}, definition.name || sectionKey),
        el("span", { class: "spacer" })
      )
    );
    const body = el("div", { class: "card-body" });
    if (sectionKey === "section_compat") {
      const covered = compatCoverage();
      body.append(
        el(
          "div",
          { class: "note", style: "margin-bottom:12px" },
          iconNode("info"),
          el(
            "span",
            {},
            covered.length
              ? `当前会被接管的标签：${covered.join("、")}`
              : "当前没有可接管的能力，框架自带那套会继续工作"
          )
        )
      );
    }
    const form = el("div", { class: "form" });
    const fields = definition.fields || {};
    const merged = new Set();
    for (const [field, item] of Object.entries(fields)) {
      if (merged.has(field)) continue;
      if (item && item.type === "info") {
        const block = el("div", { class: "field wide" });
        if (item.name) block.append(el("label", {}, item.name));
        if (item.hint || item.note) block.append(el("div", { class: "hint" }, item.hint || item.note));
        form.append(block);
        continue;
      }
      // 视频开关和白名单同排：左边开关，右边列表摘要。
      if (sectionKey === "section_features" && field === "video_enabled" && fields.video_whitelist) {
        merged.add("video_whitelist");
        form.append(
          switchListRow(
            sectionInput(sectionKey, field, item, values[field]),
            sectionInput(sectionKey, "video_whitelist", fields.video_whitelist, values.video_whitelist)
          )
        );
        continue;
      }
      if (sectionKey === "section_features" && field === "selfie_reference_files") {
        const persona = fields.selfie_use_persona;
        const companion = persona
          ? sectionInput(sectionKey, "selfie_use_persona", persona, values.selfie_use_persona)
          : null;
        merged.add("selfie_use_persona");
        form.append(referenceFilesField(item, values[field], companion));
        continue;
      }
      if (sectionKey === "section_prompts" && field === "presets_prompt") {
        form.append(inspirationBlock());
      }
      form.append(sectionInput(sectionKey, field, item, values[field]));
    }
    body.append(form);
    card.append(body);
    root.append(card);
  }
  // One request per visit, not a timer: the list only changes when the chat
  // stores or the reader deletes something.
  ensureInspirations();
}

// -- inspirations: one summary row, edited in the list modal ----------

const INSPIRATION_EMPTY =
  "还没有保存过灵感。聊天里让模型用 save_inspiration 存一条，这里就会出现。";

function inspirationCount() {
  return `共 ${Number(state.inspirations.total || 0)} 条`;
}

function inspirationBlock() {
  const row = el("div", { class: "preset-row", id: "inspirations-body" });
  const wrap = el(
    "div",
    { class: "field wide" },
    el(
      "div",
      { class: "field-head" },
      el("label", {}, "灵感库"),
      el("span", { class: "sub", id: "inspirations-count" }, inspirationCount()),
      el(
        "button",
        {
          class: "btn small ghost",
          type: "button",
          id: "inspirations-edit",
          "data-action": "inspiration-edit",
        },
        iconNode("list"),
        "管理"
      )
    ),
    row
  );
  fillInspirations(row);
  return wrap;
}

function updateInspirationActions() {
  const count = $("#inspirations-count");
  if (count) count.textContent = inspirationCount();
}

function fillInspirations(box) {
  const body = box || $("#inspirations-body");
  if (!body) return;
  updateInspirationActions();
  const data = state.inspirations;
  body.replaceChildren();
  if (!data.items.length && data.loading) {
    body.append(el("span", { class: "hint" }, "正在读取灵感…"));
    return;
  }
  if (!data.items.length && data.error) {
    body.append(el("span", { class: "hint" }, `读取灵感失败：${data.error}`));
    return;
  }
  if (!data.items.length) {
    body.append(el("span", { class: "hint" }, INSPIRATION_EMPTY));
    return;
  }
  data.items.slice(0, 3).forEach((item) => {
    const text = String(item.text || "（这条没有正文）");
    body.append(el("span", { class: "chip", title: text }, shortText(text, 18)));
  });
  if (data.items.length > 3) {
    body.append(el("span", { class: "chip chip-more" }, `+${data.items.length - 3}`));
  }
}

function ensureInspirations() {
  const data = state.inspirations;
  if (data.loaded || data.loading) return;
  loadInspirations();
}

async function loadInspirations() {
  const data = state.inspirations;
  data.loading = true;
  fillInspirations();
  try {
    const answer = await api("/inspirations");
    const items = Array.isArray(answer.items) ? answer.items : [];
    // The endpoint may answer 200 with an `error` field instead of a status
    // code; an empty list is not worth showing as "no ideas yet" when it is
    // really a read failure.
    if (answer.error && !items.length) throw new Error(String(answer.error));
    data.items = items;
    data.total = Number.isFinite(Number(answer.total)) ? Number(answer.total) : data.items.length;
    data.error = "";
  } catch (error) {
    data.error = String(error.message || error);
    data.total = data.items.length;
  } finally {
    data.loading = false;
    // Whatever happened, the card is the only thing that changes: the rest of
    // the form keeps its focus and its scroll.
    data.loaded = true;
    fillInspirations();
  }
}

// 灵感由模型在聊天里保存，这里只负责查看和删除，不给添加入口。
function openInspirationManager() {
  const modal = $("#modal");
  modal.replaceChildren();
  const card = el("div", { class: "modal-shell narrow" });
  const countChip = el("span", { class: "chip count" }, `${state.inspirations.items.length} 条`);
  card.append(
    el(
      "div",
      { class: "modal-head" },
      el("h3", { class: "modal-title" }, "灵感库"),
      countChip,
      el("span", { class: "spacer" }),
      el(
        "button",
        { class: "icon-btn", type: "button", "data-action": "modal-close", title: "关闭" },
        iconNode("close")
      )
    )
  );
  const body = el("div", { class: "modal-body" });
  const grid = el("div", { class: "entry-grid" });
  body.append(grid);

  const removeEntry = async (item, button) => {
    button.disabled = true;
    try {
      const answer = await api("/inspirations/delete", {
        method: "POST",
        body: JSON.stringify({ id: item.id }),
      });
      if (answer && answer.ok === false) throw new Error(answer.error || "删除没成功");
      state.inspirations.items = state.inspirations.items.filter((entry) => entry.id !== item.id);
      state.inspirations.total = state.inspirations.items.length;
      paint();
      fillInspirations();
    } catch (error) {
      toast(String(error.message || error), "error");
      button.disabled = false;
    }
  };

  const promoteEntry = (item) => {
    const text = String(item.text || "");
    const nameInput = el("input", {
      class: "input",
      type: "text",
      maxlength: "24",
      value: text.slice(0, 10),
    });
    const targetSelect = el(
      "select",
      { class: "input" },
      el("option", { value: "draw" }, "文生图预设"),
      el("option", { value: "edit" }, "改图预设")
    );
    const back = el("button", { class: "btn ghost", type: "button" }, "返回");
    back.addEventListener("click", () => openInspirationManager());
    const go = el("button", { class: "btn primary", type: "button" }, "收进预设");
    go.addEventListener("click", async () => {
      go.disabled = true;
      try {
        const answer = await api("/inspirations/promote", {
          method: "POST",
          body: JSON.stringify({
            id: item.id,
            name: nameInput.value,
            target: targetSelect.value,
          }),
        });
        if (answer && answer.ok === false) throw new Error(answer.error || "没收进去");
        applyConfig(await api("/config?lite=1"));
        state.inspirations.loaded = false;
        await loadInspirations();
        closeModal();
        render();
        toast("已收进预设", "ok");
      } catch (error) {
        toast(String(error.message || error), "error");
        go.disabled = false;
      }
    });
    body.replaceChildren(
      el("div", { class: "modal-note" }, `把这条灵感收进预设：${shortText(text, 40)}`),
      el(
        "div",
        { class: "field" },
        el("label", {}, "预设名字"),
        nameInput
      ),
      el(
        "div",
        { class: "field" },
        el("label", {}, "收到哪里"),
        targetSelect
      ),
      el("div", { class: "modal-actions" }, back, go)
    );
  };

  const paint = () => {
    const items = state.inspirations.items;
    countChip.textContent = `${items.length} 条`;
    grid.replaceChildren();
    if (!items.length) {
      grid.append(
        el(
          "div",
          { class: "entry-hollow" },
          el("span", { class: "entry-hollow-title" }, "还没有保存过灵感"),
          el(
            "span",
            { class: "entry-hollow-tip" },
            "聊天里让模型用 save_inspiration 存一条，这里就能看到；这里只能查看和删除。"
          )
        )
      );
      return;
    }
    items.forEach((item, index) => {
      const text = String(item.text || "（这条没有正文）");
      const row = el("div", { class: "entry" });
      row.append(el("span", { class: "entry-no" }, String(index + 1).padStart(2, "0")));
      row.append(el("span", { class: "entry-text", title: text }, text));
      if (item.promoted) row.append(el("span", { class: "chip wait" }, "已收录"));
      const push = el(
        "button",
        { class: "entry-drop", type: "button", title: "收进预设" },
        iconNode("save")
      );
      push.addEventListener("click", () => promoteEntry(item));
      row.append(push);
      const drop = el(
        "button",
        { class: "entry-drop", type: "button", title: "删除这条灵感" },
        iconNode("trash")
      );
      drop.addEventListener("click", () => void removeEntry(item, drop));
      row.append(drop);
      grid.append(row);
    });
  };

  body.append(
    el(
      "div",
      { class: "modal-actions" },
      el("span", { class: "hint" }, "灵感由模型保存，这里只能删除。"),
      el("span", { class: "spacer" }),
      el("button", { class: "btn small ghost", type: "button", "data-action": "modal-close" }, "关闭")
    )
  );
  card.append(body);
  modal.append(card);
  modal.classList.remove("hidden");
  card.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    event.stopPropagation();
    closeModal();
  });
  paint();
  card.querySelector("[data-action='modal-close']")?.focus();
}

// -- providers: mutations --------------------------------------------

function selectedProvider() {
  return (state.config.providers || []).find((item) => item.id === state.selected) || null;
}

function addProvider() {
  const providers = state.config.providers;
  const template = state.templates[0];
  if (!template) return;
  let index = providers.length + 1;
  let id = `provider${index}`;
  while (providers.some((item) => item.id === id)) {
    index += 1;
    id = `provider${index}`;
  }
  const defaults = structuredClone(state.providerDefaults[template.key] || {});
  providers.push({ ...defaults, id, template_key: template.key });
  state.selected = id;
  state.providerError = "";
  render();
}

function deleteProvider() {
  const provider = selectedProvider();
  if (!provider) return;
  if (!window.confirm(`删除服务商「${provider.id}」？链路里引用它的条目也会一起移除。`)) {
    return;
  }
  state.config.providers = state.config.providers.filter((item) => item.id !== provider.id);
  for (const [feature, chain] of Object.entries(state.config.chains || {})) {
    state.config.chains[feature] = (chain || []).filter((item) => item.provider !== provider.id);
  }
  state.selected = state.config.providers[0]?.id || null;
  render();
}

function renameProvider(nextId) {
  const provider = selectedProvider();
  if (!provider) return;
  const clean = String(nextId || "").trim();
  if (!clean || clean === provider.id) {
    render();
    return;
  }
  if ((state.config.providers || []).some((item) => item.id === clean)) {
    toast(`已经有叫「${clean}」的服务商了`, "error");
    render();
    return;
  }
  const previous = provider.id;
  provider.id = clean;
  for (const [feature, chain] of Object.entries(state.config.chains || {})) {
    state.config.chains[feature] = (chain || []).map((item) =>
      item.provider === previous ? { ...item, provider: clean } : item
    );
  }
  state.selected = clean;
  render();
}

// -- events -----------------------------------------------------------

function onViewClick(event) {
  const button = event.target.closest("[data-action]");
  if (button) {
    const action = button.dataset.action;
    if (action === "goto") {
      location.hash = `#${button.dataset.view}`;
      return;
    }
    if (action === "provider-tab") {
      state.providerTab = button.dataset.tab;
      render();
      return;
    }
    if (action === "boot-retry") {
      boot();
      return;
    }
    if (action === "provider-add") {
      addProvider();
      return;
    }
    if (action === "provider-delete") {
      deleteProvider();
      return;
    }
    if (action === "chain-move") {
      const feature = button.dataset.feature;
      const index = Number(button.dataset.index);
      const delta = Number(button.dataset.delta);
      const chain = state.config.chains?.[feature] || [];
      const next = index + delta;
      if (next < 0 || next >= chain.length) return;
      [chain[index], chain[next]] = [chain[next], chain[index]];
      render();
      return;
    }
    if (action === "chain-remove") {
      const feature = button.dataset.feature;
      const index = Number(button.dataset.index);
      state.config.chains[feature].splice(index, 1);
      render();
      return;
    }
    if (action === "task-cancel") {
      cancelTask(button.dataset.task);
      return;
    }
    if (action === "tasks-refresh") {
      loadTasks();
      refreshStatus();
      return;
    }
    if (action === "fetch-models") {
      fetchModels();
      return;
    }
    if (action === "list-edit") {
      openListEditorFor(button.dataset.listTarget || "", button.dataset.listTitle || "");
      return;
    }
    if (action === "gallery-tab") {
      state.galleryTab = button.dataset.tab || "all";
      state.galleryPicked = [];
      render();
      return;
    }
    if (action === "gallery-select") {
      state.gallerySelect = !state.gallerySelect;
      state.galleryPicked = [];
      render();
      return;
    }
    if (action === "gallery-pick") {
      applyGalleryPick(button.closest(".tile"), button.dataset.image);
      return;
    }
    if (action === "gallery-pick-all") {
      selectAllGallery();
      return;
    }
    if (action === "bulk-favorite") {
      bulkGallery(state.galleryTab === "fav" ? "unfavorite" : "favorite");
      return;
    }
    if (action === "bulk-delete") {
      bulkGallery("delete");
      return;
    }
    if (action === "bulk-clear") {
      state.galleryPicked = [];
      render();
      return;
    }
    if (action === "task-filter") {
      filterTasks(button.dataset.state || "");
      return;
    }
    if (action === "task-retry") {
      retryTask(button.dataset.task);
      return;
    }
    if (action === "gallery-refresh") {
      loadGallery(state.galleryPage);
      return;
    }
    if (action === "gallery-page") {
      loadGallery(Number(button.dataset.page) || 1);
      return;
    }
    if (action === "gallery-favorite" || action === "image-favorite") {
      favoriteImage(button.dataset.image, button.dataset.favorite !== "1");
      return;
    }
    if (action === "refs-manage") {
      loadReferenceFiles().then(() => openReferenceManager());
      return;
    }
    if (action === "presets-edit") {
      openPresetEditor(button.dataset.section, button.dataset.sectionField);
      return;
    }
    if (action === "inspiration-edit") {
      openInspirationManager();
      return;
    }
    if (action === "gallery-delete" || action === "image-delete") {
      deleteImage(button.dataset.image);
      return;
    }
    if (action === "image-download") {
      downloadImage(button.dataset.image);
      return;
    }
    if (action === "modal-close") {
      closeModal();
      return;
    }
  }
  const shot = event.target.closest(".tile .shot");
  if (shot) {
    const tile = shot.closest(".tile");
    if (state.gallerySelect) applyGalleryPick(tile, tile.dataset.image);
    else if (shot.dataset.imageId) openImage(shot.dataset.imageId);
  }
}

function onViewChange(event) {
  const target = event.target;
  if (target.matches("[data-provider-id='1']")) {
    renameProvider(target.value);
    return;
  }
  if (target.matches("[data-provider-id='template']")) {
    const provider = selectedProvider();
    if (!provider || target.value === provider.template_key) return;
    const defaults = structuredClone(state.providerDefaults[target.value] || {});
    const keptId = provider.id;
    const nextTemplate = state.templates.find((item) => item.key === target.value);
    const sameKind =
      nextTemplate?.kind ===
      state.templates.find((item) => item.key === provider.template_key)?.kind;
    const keeps = {};
    for (const key of ["id", "api_keys", "base_url", "proxy_url"]) {
      if (provider[key]) keeps[key] = provider[key];
    }
    if (sameKind && provider.model) keeps.model = provider.model;
    Object.keys(provider).forEach((key) => delete provider[key]);
    Object.assign(provider, defaults, keeps, { id: keptId, template_key: target.value });
    render();
    return;
  }
  if (target.matches("[data-provider-field]")) {
    const provider = selectedProvider();
    if (!provider) return;
    const field = target.dataset.providerField;
    const kind = target.dataset.kind;
    provider[field] = readControl(target, kind);
    if (field === "supports_edit") {
      const text = target.parentElement.querySelector(".track + span");
      if (text) text.textContent = provider[field] ? "已开启" : "已关闭";
    }
    updateTopbar();
    const dot = $('.provider-row[data-selected="true"] .dot');
    if (dot) dot.className = `dot ${providerComplete(provider) ? "ok" : "warn"}`;
    return;
  }
  if (target.matches("[data-chain-field]")) {
    const feature = target.dataset.chain;
    const index = Number(target.dataset.chainIndex);
    const chain = state.config.chains?.[feature];
    if (!chain || !chain[index]) return;
    chain[index][target.dataset.chainField] = target.value;
    updateTopbar();
    return;
  }
  if (target.matches("[data-chain-add]")) {
    const feature = target.dataset.chainAdd;
    if (!target.value) return;
    state.config.chains[feature] = state.config.chains[feature] || [];
    state.config.chains[feature].push({ provider: target.value, output: "" });
    render();
    return;
  }
  if (target.matches("[data-section-field]")) {
    const section = target.dataset.section;
    const field = target.dataset.sectionField;
    state.config[section] = state.config[section] || {};
    state.config[section][field] = readControl(target, target.dataset.kind);
    if (target.dataset.kind === "bool") {
      const text = target.parentElement.querySelector(".switch-text");
      if (text) text.textContent = state.config[section][field] ? "已开启" : "已关闭";
    }
    if (section === "section_compat") render();
    else updateTopbar();
    if (section === "section_features") updateTopbar();
    return;
  }
}

function onViewInput(event) {
  const target = event.target;
  if (
    target.matches("[data-section-field][data-kind='int'], [data-section-field][data-kind='float']") ||
    target.matches("[data-provider-field][data-kind='int']")
  ) {
    onViewChange(event);
  }
}

function readControl(control, kind) {
  if (kind === "bool") return control.checked;
  if (kind === "int") {
    const value = Number.parseInt(control.value, 10);
    return Number.isNaN(value) ? 0 : value;
  }
  if (kind === "float") {
    const value = Number.parseFloat(control.value);
    return Number.isNaN(value) ? 0 : value;
  }
  if (kind === "list") {
    return control.value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
  }
  if (kind === "keys") {
    return control.value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
  }
  if (kind === "json") {
    const text = control.value.trim();
    if (!text) return {};
    try {
      const parsed = JSON.parse(text);
      control.style.borderColor = "";
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      control.style.borderColor = "var(--danger)";
      return control.__lastGood || {};
    }
  }
  return control.value;
}

function closeModal() {
  const modal = $("#modal");
  const dismiss = modalDismiss;
  modalDismiss = null;
  modal.classList.add("hidden");
  modal.replaceChildren();
  // A dialog closed from the outside still has to give its caller an answer,
  // otherwise whoever awaited it waits forever.
  if (dismiss) dismiss();
}

async function cancelTask(taskId) {
  try {
    const data = await api("/tasks/cancel", {
      method: "POST",
      body: JSON.stringify({ task_id: taskId }),
    });
    toast(data.ok ? "已取消" : "取消没成功，可能刚开始跑");
    loadTasks();
    refreshStatus();
  } catch (error) {
    toast(String(error.message || error), "error");
  }
}

async function retryTask(taskId) {
  if (!taskId) return;
  try {
    const data = await api(`/tasks/${encodeURIComponent(taskId)}/retry`, {
      method: "POST",
      body: JSON.stringify({ task_id: taskId }),
    });
    toast(data.ok ? "已重新入队" : data.error || "重试没成功", data.ok ? "info" : "error");
    if (data.ok) {
      await loadTasks();
      refreshStatus();
    }
  } catch (error) {
    toast(String(error.message || error), "error");
  }
}

function filterTasks(nextState) {
  state.taskState = nextState || "";
  render();
}

async function favoriteImage(imageId, favorite) {
  try {
    await api("/gallery/favorite", {
      method: "POST",
      body: JSON.stringify({ image_id: imageId, favorite }),
    });
    const item = (state.gallery || []).find((row) => row.id === imageId);
    if (item) item.favorite = Boolean(favorite);
    const modalButton = document.querySelector(
      `#modal [data-action="image-favorite"][data-image="${imageId}"]`
    );
    if (modalButton) {
      modalButton.dataset.favorite = favorite ? "1" : "0";
      modalButton.classList.toggle("is-on", Boolean(favorite));
      modalButton.replaceChildren(
        iconNode("star"),
        document.createTextNode(favorite ? "取消收藏" : "收藏")
      );
    }
    const tileButton = document.querySelector(
      `#view [data-action="gallery-favorite"][data-image="${imageId}"]`
    );
    if (tileButton) {
      tileButton.dataset.favorite = favorite ? "1" : "0";
      tileButton.dataset.on = favorite ? "true" : "false";
      tileButton.title = favorite ? "取消收藏" : "收藏";
    }
    const countLabel = $("#gallery-count");
    if (countLabel) countLabel.textContent = gallerySummary();
    refreshStatus();
  } catch (error) {
    toast(String(error.message || error), "error");
  }
}

async function deleteImage(imageId) {
  if (!window.confirm("删除这张图片？文件也会一起删掉。")) return;
  try {
    await api("/gallery/delete", {
      method: "POST",
      body: JSON.stringify({ image_id: imageId }),
    });
    toast("已删除");
    await loadGallery();
    refreshStatus();
  } catch (error) {
    toast(String(error.message || error), "error");
  }
}

async function bulkGallery(action) {
  const ids = [...state.galleryPicked];
  if (!ids.length) return;
  if (
    action === "delete" &&
    !window.confirm(`删除选中的 ${ids.length} 张图片？文件也会一起删掉。`)
  ) {
    return;
  }
  try {
    const data = await api("/gallery/bulk", {
      method: "POST",
      body: JSON.stringify({ action, ids }),
    });
    if (!data.ok) throw new Error(data.error || "操作没有成功");
    if (action === "delete") {
      const skipped = Number(data.skipped || 0);
      toast(
        skipped
          ? `已删除 ${data.deleted} 张，${skipped} 张收藏未删除`
          : `已删除 ${data.deleted} 张`
      );
    } else if (action === "unfavorite") {
      toast(`已取消收藏 ${data.updated || 0} 张`);
    } else {
      toast(`已收藏 ${data.updated || 0} 张`);
    }
    state.galleryPicked = [];
    await loadGallery();
    refreshStatus();
  } catch (error) {
    toast(String(error.message || error), "error");
  }
}

// -- boot -------------------------------------------------------------

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const button = $("#theme-toggle");
  button.replaceChildren(iconNode(theme === "dark" ? "sun" : "moon"));
  button.title = theme === "dark" ? "换成浅色" : "换成深色";
  localStorage.setItem(THEME_KEY, theme);
}

let navBusy = false;
// Set while a dialog is waiting for an answer. Any way of closing the modal
// (Esc, the X, a click on the backdrop) runs it, so the caller never hangs.
let modalDismiss = null;

function viewFromHash() {
  const name = (location.hash || "").replace("#", "");
  return VIEWS[name] ? name : "overview";
}

// One gate for everything that would drop unsaved edits: saving or discarding
// happens here, and "stay" means the caller must not continue.
async function leaveGuard() {
  if (!dirty()) return true;
  navBusy = true;
  try {
    const choice = await askUnsaved();
    if (choice === "stay") return false;
    if (choice === "save") return await save(false);
    state.config = structuredClone(state.original);
    return true;
  } finally {
    navBusy = false;
  }
}

async function route() {
  const next = viewFromHash();
  if (next === state.view) {
    render();
    return;
  }
  if (navBusy) return;
  if (!(await leaveGuard())) {
    if (state.view) location.hash = `#${state.view}`;
    return;
  }
  if (state.view === "settings" && next !== "settings") {
    // Next visit reads the list again; until then the old copy is not reused.
    state.inspirations.loaded = false;
  }
  state.view = next;
  render();
  // A second click during the dialog changes the hash but not the view; putting
  // the hash back keeps the address bar and the page telling the same story.
  if (location.hash !== `#${next}`) location.hash = `#${next}`;
}

function askUnsaved() {
  return new Promise((resolve) => {
    const modal = $("#modal");
    modal.replaceChildren();
    let settled = false;
    const finish = (choice) => {
      if (settled) return;
      settled = true;
      modalDismiss = null;
      closeModal();
      resolve(choice);
    };
    // Esc, the X and the backdrop all mean "cancel": nothing is lost and the
    // navigation that opened this dialog simply does not happen.
    modalDismiss = () => finish("stay");
    const card = el("div", { class: "modal-card unsaved-card" });
    card.append(
      el(
        "div",
        { class: "modal-head" },
        el("h3", { class: "modal-title" }, "还没保存"),
        el("span", { class: "spacer" }),
        el(
          "button",
          { class: "icon-btn", type: "button", "data-unsaved": "stay", title: "关闭" },
          iconNode("close")
        )
      ),
      el("p", { class: "modal-text" }, "当前页面有还没保存的修改，要先保存吗？"),
      el(
        "div",
        { class: "modal-actions" },
        el(
          "button",
          { class: "btn ghost", type: "button", "data-unsaved": "discard" },
          "不保存"
        ),
        el("button", { class: "btn ghost", type: "button", "data-unsaved": "stay" }, "取消"),
        el("button", { class: "btn primary", type: "button", "data-unsaved": "save" }, "保存")
      )
    );
    card.addEventListener("click", (event) => {
      const button = event.target.closest("[data-unsaved]");
      if (button) finish(button.dataset.unsaved);
    });
    modal.replaceChildren(card);
    modal.classList.remove("hidden");
  });
}

function bind() {
  $("#nav").addEventListener("click", (event) => {
    const button = event.target.closest("[data-nav]");
    if (button) location.hash = `#${button.dataset.nav}`;
  });
  $("#view").addEventListener("click", onViewClick);
  $("#view").addEventListener("change", onViewChange);
  $("#view").addEventListener("input", onViewInput);
  $("#btn-save").addEventListener("click", () => save(false));
  $("#btn-discard").addEventListener("click", () => {
    if (!dirty()) return;
    if (!window.confirm("放弃这次修改？")) return;
    state.config = structuredClone(state.original);
    render();
  });
  $("#btn-refresh").addEventListener("click", async () => {
    // Reloading also drops whatever is only in the form, so it goes through
    // the same save-or-not question as leaving the page.
    if (navBusy) return;
    if (!(await leaveGuard())) return;
    load().catch((error) => toast(String(error.message || error), "error"));
  });
  $("#theme-toggle").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
  $("#modal").addEventListener("click", async (event) => {
    if (event.target.id === "modal") {
      closeModal();
      return;
    }
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const action = button.dataset.action;
    if (action === "modal-close") {
      closeModal();
      return;
    }
    if (action === "image-favorite") {
      await favoriteImage(button.dataset.image, button.dataset.favorite !== "1");
      return;
    }
    if (action === "image-delete") {
      const imageId = button.dataset.image;
      await deleteImage(imageId);
      closeImageModal();
      return;
    }
    if (action === "image-download") {
      await downloadImage(button.dataset.image);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const modal = $("#modal");
    if (!modal || modal.classList.contains("hidden")) return;
    event.preventDefault();
    // Closing this way answers a pending dialog too, so navBusy is released.
    closeModal();
  });
  window.addEventListener("beforeunload", (event) => {
    if (!dirty()) return;
    event.preventDefault();
    // Older browsers still need a value here to show their own wording.
    event.returnValue = "";
  });
  window.addEventListener("hashchange", route);
}

applyTheme(localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light");
bind();
boot();

// The first screen must not accuse the plugin before it has had a chance to
// answer: the hash picks the view, a skeleton fills the page, and only a load
// that still fails after one quiet retry becomes the error card.
async function boot() {
  state.booting = true;
  state.view = viewFromHash();
  render();
  const ok = await start();
  state.booting = false;
  if (ok) {
    route();
    return;
  }
  render();
  toast(state.loadError, "error");
}

async function start() {
  try {
    await load();
    return true;
  } catch (error) {
    // A restarted framework or plugin can miss the first request; retry once quietly.
    await new Promise((resolve) => setTimeout(resolve, 900));
    try {
      await load();
      return true;
    } catch (retryError) {
      state.config = null;
      state.loadError = String(retryError.message || retryError);
      return false;
    }
  }
}
