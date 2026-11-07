"""Static web assets must parse, and the page must reference real files."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Older node builds refuse to syntax-check an .mjs file ("Cannot use import
# statement outside a module"); this helper parses it as a real module instead.
_VM_CHECKER = """
const fs = require('fs');
const vm = require('vm');
let failed = false;
for (const file of process.argv.slice(2)) {
  try {
    new vm.SourceTextModule(fs.readFileSync(file, 'utf8'), { identifier: file });
  } catch (err) {
    console.error(`${file}: ${err.message}`);
    failed = true;
  }
}
process.exit(failed ? 1 : 0);
"""


def _node():
    return shutil.which("node")


def _parse_error(node: str, target: Path, tmp_path: Path) -> str:
    """Return the parse error for a script, or an empty string when it is fine."""
    result = subprocess.run(
        [node, "--check", str(target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        return ""
    # The page scripts are ES modules; node's CJS syntax check rejects those.
    helper = tmp_path / "check-module.cjs"
    helper.write_text(_VM_CHECKER, encoding="utf-8")
    module = subprocess.run(
        [node, "--experimental-vm-modules", str(helper), str(target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if module.returncode == 0:
        return ""
    return (module.stderr.strip() or module.stdout.strip() or result.stderr.strip())


def test_scripts_parse(tmp_path):
    node = _node()
    if not node:
        pytest.skip("node is not available")
    for name in ("app.js", "icons.js"):
        source = WEB_DIR / name
        target = tmp_path / f"{source.stem}-check.mjs"
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        error = _parse_error(node, target, tmp_path)
        assert not error, f"{name}: {error}"


def test_page_references_existing_assets():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    for name in ("app.js", "app.css", "icons.js", "icon.png"):
        assert (WEB_DIR / name).is_file(), f"missing asset: {name}"
    assert "./app.js" in html
    assert "./app.css" in html


def test_gallery_uses_binary_endpoints():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "/gallery/thumb?image_id=" in script
    assert "/gallery/preview?image_id=" in script
    assert "/gallery/file?image_id=" in script
    assert "/gallery/image?image_id=" not in script
    # The viewer opens with the preview; the original is only a last resort.
    assert 'picture.src = `${BASE}/gallery/preview' in script


def test_reference_row_pairs_the_upload_with_the_persona_switch():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    block = script.split("const persona = fields.selfie_use_persona;")[1].split("continue;")[0]

    assert "selfie_use_persona" in block
    assert "selfie_follow_edit_chain" not in block
    assert "referenceFilesField(item, values[field], companion)" in block


def test_provider_rows_show_health_from_the_metrics_endpoint():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")

    assert 'api("/metrics")' in script
    assert "function healthText(row)" in script
    assert "冷却 ${Math.ceil(row.cooling_seconds)} 秒" in script
    assert "成功率 ${percent}%" in script
    # The providers page marks the open row; the overview card must stay flat.
    assert "function providerRow(provider, template, onClick, highlight = false)" in script
    assert '.provider-row[data-selected="true"] {' in css
    overview = script.split("const providerCard")[1].split("providerCard.append(body)")[0]
    assert "}, true)" not in overview
    page = script.split("split.append(list, renderProviderEditor())")[0].split("function renderProviders")[1]
    assert "}, true)" in page


def test_reference_edits_redraw_the_open_manager():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    helper = script.split("function applyReferenceFiles(files)")[1].split("\nasync function deleteReference")[0]

    assert "render();" in helper
    assert "openReferenceManager();" in helper

    upload = script.split("async function uploadReferences(fileList)")[1].split("function applyReferenceFiles")[0]
    delete = script.split("async function deleteReference(name)")[1].split("function openPresetEditor")[0]
    assert "applyReferenceFiles(data.files)" in upload
    assert "applyReferenceFiles(data?.files || [])" in delete


def test_reference_cards_offer_roles_and_ordering():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "const REFERENCE_ROLE_LABELS" in script
    for role, label in (
        ("subject", "主体"),
        ("clothing", "服装"),
        ("style", "画风"),
        ("pose", "姿势"),
        ("background", "背景"),
    ):
        assert f'{role}: "{label}"' in script
    assert "/files/role" in script
    assert "/files/order" in script
    assert "setReferenceRole" in script
    assert "moveReference" in script
    assert "上移" in script
    assert "下移" in script
    assert "ref-role" in script
    assert "refs-up" in script
    assert "refs-down" in script


def test_reference_controls_are_styled():
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")

    for selector in (".ref-thumb", ".ref-controls", ".ref-role", ".ref-move"):
        assert selector in css


def test_gallery_toolbar_searches_filters_and_selects():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    # Debounced prompt search plus kind/sort selects in the gallery toolbar.
    assert 'class: "gallery-search"' in script
    assert "搜索提示词" in script
    assert '"", "全部类型"' in script
    assert '"image", "图片"' in script
    assert '"video", "视频"' in script
    assert '"new", "最新"' in script
    assert '"old", "最早"' in script
    assert "window.setTimeout(apply, 300)" in script

    # The query string carries the filters; favorites tab is server-side.
    assert 'params.set("q", state.galleryQuery)' in script
    assert 'params.set("kind", state.galleryKind)' in script
    assert 'params.set("favorite", "1")' in script
    assert 'params.set("sort", state.gallerySort)' in script
    assert 'data.items' in script

    # Multi-select mode and the bulk bar.
    assert 'data-action": "gallery-select"' in script
    assert "退出多选" in script
    assert 'class: "gallery-bulk"' in script
    assert '"data-action": "bulk-delete"' in script
    assert '"data-action": "bulk-favorite"' in script
    assert "已选" in script
    assert 'api("/gallery/bulk"' in script
    assert "JSON.stringify({ action, ids })" in script


def test_image_modal_loads_details_from_the_item_endpoint():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert 'class: "modal-facts"' in script
    assert 'id: "image-facts"' in script
    assert "/gallery/item/${" in script
    for label in ("服务商", "模型", "尺寸", "类型", "创建时间", "收藏", "文件大小", "文件名"):
        assert f'["{label}"' in script


def test_gallery_pages_by_sixty_with_a_jump_box():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")

    assert "const GALLERY_PAGE = 60;" in script
    # Real pages, not an append-only feed.
    assert "loadGallery(append" not in script
    assert "gallery-more" not in script
    assert 'offset: String((wanted - 1) * GALLERY_PAGE)' in script

    assert "function galleryPager(pages)" in script
    assert "上一页" in script and "下一页" in script
    assert "function gotoGalleryPage(value, pages)" in script
    # The page number is an editable box followed by "/ total".
    assert 'class: "pager-jump"' in script
    assert 'class: "pager-total"' in script
    assert "`/ ${pages}`" in script
    assert "当前页码，输入后回车跳转" in script
    assert '"chevron-left"' in script and '"chevron-right"' in script

    for selector in (".gallery-pager", ".pager-total", ".pager-jump", ".pager-nav"):
        assert selector in css


def test_tasks_page_is_a_single_line_list():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")

    assert "function taskList(tasks)" in script
    assert 'class: "task-line"' in script
    assert "function taskStatusCell(task)" in script
    assert "task-line-progress" in script
    assert "只读汇总，不会改动队列" in script
    # The page is read-only: retry and cancel belong to the conversation.
    assert "task-retry" not in script
    assert "task-cancel" not in script
    assert "/retry" not in script
    assert "TASK_FILTERS" not in script
    assert ".task-line" in css
    assert ".task-table" not in css


def test_api_keys_use_the_list_item_editor():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    icons = (WEB_DIR / "icons.js").read_text(encoding="utf-8")

    assert "function providerKeysField(provider)" in script
    assert "return providerKeysField(provider);" in script
    assert 'title: `${fieldLabel("api_keys")}列表`' in script
    # Opening the key editor asks for the full keys; the page itself only has masks.
    assert "async function providerFullKeys(providerId)" in script
    assert 'api("/provider_keys"' in script
    assert "items = await providerFullKeys(provider.id);" in script
    # The editor adds one item per Enter press and can take a pasted block.
    assert "function openListEditor(options)" in script
    assert "function openListImport(options)" in script
    assert "输入一项后按回车添加" in script
    assert "输入密钥后按回车添加" in script
    assert "空行会自动跳过" in script
    # Provider keys are editable secret fields with a show/hide toggle.
    assert 'secret: field === "api_keys"' in script
    assert "const secret = Boolean(options.secret);" in script
    assert 'type: secret ? "password" : "text"' in script
    assert "const secretEntry = (text, index) => {" in script
    assert 'class: "secret-input"' in script
    assert 'class: "secret-toggle"' in script
    assert 'title: "显示密钥"' in script
    assert 'toggle.replaceChildren(iconNode(hidden ? "eye-off" : "eye"));' in script
    assert "eye: [" in icons
    assert '"eye-off": [' in icons
    assert 'if (action === "list-edit")' in script
    assert "if (!draft.includes(text)) draft.push(text);" in script
    # Enter keeps the caret in the box, drops the row into view and trims duplicates.
    assert "addInput.focus();" in script
    assert "grid.scrollTop = grid.scrollHeight;" in script
    assert "function normalizeListItems(items)" in script
    # Escape closes the dialog, Ctrl / Cmd + Enter commits the pasted block.
    assert 'event.key !== "Escape"' in script
    assert "event.ctrlKey || event.metaKey" in script
    assert 'title: "删除这一项"' in script
    for selector in (
        ".compose",
        ".compose-input",
        ".compose-send",
        ".entry-grid",
        ".entry",
        ".entry-no",
        ".entry-text",
        ".entry-drop",
        ".entry-secret",
        ".secret-field",
        ".secret-input",
        ".secret-toggle",
        ".entry-hollow",
        ".paste-shell",
        ".paste-head",
        ".paste-count",
        ".paste-area",
    ):
        assert selector in css

    # The scrollbar gets its own gutter so it never sits on top of the text.
    assert "scrollbar-gutter: stable" in css


def test_settings_form_tightens_numbers_and_autosizes_textareas():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")

    # Number and dropdown knobs stack their value right under the label.
    assert 'const compact = kind === "integer" || kind === "float" || kind === "select";' in script
    assert '${compact ? " compact" : ""}' in script
    assert 'class: `field${wide ? " wide" : ""}' in script
    # Textareas grow with their content between a floor and a ceiling.
    assert "const TEXTAREA_MIN = 74;" in script
    assert "const TEXTAREA_MAX = 320;" in script
    assert "function fitTextarea(area)" in script
    assert "function fitTextareas(root)" in script
    assert '"data-autosize": "1"' in script
    assert "fitTextareas(view);" in script

    for selector in (
        ".field.compact",
        '.field.compact input[type="number"]',
        ".field.compact select",
        ".field textarea.grow",
    ):
        assert selector in css
    assert "flex-direction: row" not in css.split(".field.compact {")[1].split("}")[0]


def test_access_whitelist_uses_the_list_item_editor():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert 'if (sectionKey === "section_access" && field === "whitelist")' in script
    assert "return listField(sectionKey, field, item, value);" in script
    assert 'state.config[sectionKey][field] = draft.join("\\n");' in script
    # The dialog is titled after the field, so the whitelist reads「白名单列表」.
    assert 'title: label.endsWith("列表") ? label : `${label}列表`' in script
    assert '"data-list-title": options.title' in script


def test_result_origin_allowlist_uses_the_list_item_editor():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert (
        'if (sectionKey === "section_advanced" && field === "trusted_result_origins")'
        in script
    )


def test_preset_editor_button_sits_next_to_the_chip():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    start = script.index("function presetField(")
    body = script[start : script.index("function ", start + 10)]

    # No spacer: the button follows the chip inside the row.
    assert 'el("span", { class: "spacer" })' not in body
    assert "chip-add" in body
    assert body.count("row.append(") >= 2
    assert "presets-edit" in body


def test_advanced_section_exposes_network_knobs():
    schema = json.loads((WEB_DIR.parent / "schema.json").read_text(encoding="utf-8"))
    fields = schema["section_advanced"]["fields"]

    for name in (
        "download_timeout_seconds",
        "image_max_mb",
        "video_max_mb",
        "trusted_result_origins",
        "allow_private_result_origins",
    ):
        assert name in fields, f"missing advanced field: {name}"
    assert schema["section_advanced"]["collapsed"] is True
    for name in ("thumb_width", "thumb_format", "preview_size", "preview_format"):
        assert name not in fields, f"{name} belongs to the gallery section"


def test_gallery_section_owns_the_encoding_knobs():
    schema = json.loads((WEB_DIR.parent / "schema.json").read_text(encoding="utf-8"))
    fields = schema["section_gallery"]["fields"]

    for name in (
        "thumb_width",
        "thumb_quality",
        "thumb_format",
        "preview_size",
        "preview_quality",
        "preview_format",
    ):
        assert name in fields, f"missing gallery field: {name}"
    assert fields["thumb_format"]["type"] == "select"
    assert fields["thumb_format"]["options"] == ["webp", "jpeg"]
    assert fields["preview_format"]["type"] == "select"
    assert fields["preview_format"]["options"] == ["jpeg", "webp"]


def test_settings_page_renders_select_fields_as_dropdowns():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert 'kind === "select"' in script
    assert '"data-section-field": field' in script


def test_video_whitelist_uses_the_list_item_editor():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert 'if (sectionKey === "section_features" && field === "video_whitelist")' in script
    branch = script.split('field === "video_whitelist"')[1].split("if (sectionKey")[0]
    assert "return listField(sectionKey, field, item, value);" in branch
    # 列表摘要上的按钮走 list-edit，最后仍然落到同一个列表编辑器。
    assert "function listField(sectionKey, field, item, value)" in script
    assert 'data-action": "list-edit"' in script
    assert "target: `section:${sectionKey}:${field}`" in script
    assert "function openListEditorFor(target, title)" in script
    opener = script.split("function openListEditorFor(target, title)")[1].split(
        "function openListEditor(options)"
    )[0]
    assert "openListEditor({" in opener
    assert "function openListEditor(options)" in script


def test_video_row_puts_the_list_summary_beside_the_switch():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")

    # 开关这一行自己开 field-row，白名单从正常顺序里跳过，不会重复渲染。
    row = script.split('field === "video_enabled"')[1].split("continue;")[0]
    assert "fields.video_whitelist" in row
    assert "merged.add(\"video_whitelist\");" in row
    assert "function switchListRow(switchField, listSummary)" in script
    assert 'class: "field-row feature-row"' in script

    # 两列各自贴顶；列表摘要只占自己那一列，标签多了在本列内换行。
    rule = css.split(".field-row.feature-row {")[1].split("}")[0]
    assert "align-items: start;" in rule
    narrow = css.split(".field-row.feature-row > .field.wide {")[1].split("}")[0]
    assert "grid-column: auto;" in narrow
    # 复用已有的 chip 换行容器。
    assert "flex-wrap: wrap;" in css.split(".preset-row {")[1].split("}")[0]

