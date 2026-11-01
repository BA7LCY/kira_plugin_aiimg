"""Static web assets must parse, and the page must reference real files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _node():
    return shutil.which("node")


def test_scripts_parse(tmp_path):
    node = _node()
    if not node:
        pytest.skip("node is not available")
    for name in ("app.js", "icons.js"):
        source = WEB_DIR / name
        target = tmp_path / f"{source.stem}-check.mjs"
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        result = subprocess.run(
            [node, "--check", str(target)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"{name}: {result.stderr.strip()}"


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


def test_tasks_page_filters_and_retries_failed_jobs():
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    filters = script.split("const TASK_FILTERS = [")[1].split("];")[0]
    for label in ("全部", "运行中", "已完成", "失败", "已取消"):
        assert f'"{label}"' in filters
    assert '"data-action": "task-filter"' in script
    assert 'params.set("state", state.taskState)' in script

    assert '"data-action": "task-retry"' in script
    assert "/tasks/${encodeURIComponent(taskId)}/retry" in script
    assert "已重新入队" in script

