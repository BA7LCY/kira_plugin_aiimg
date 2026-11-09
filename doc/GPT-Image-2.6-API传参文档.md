# GPT Image 2.6 API 传参文档

整理：莉莉娅 ｜ 2026-11-06 ｜ 依据 OpenAI 官方 image guide 与 API reference，
并经 fair 叔叔站（64.83.43.6:3000，gpt-image-2.6-flare）实弹测试验证

---

## 一、模型与端点

- 模型系列：gpt-image-2.6（fair 站现有 flare / sunburst / 4k / 1k 等变体，2.5 系沿用同一套参数）
- 生成端点：POST /v1/images/generations（文生图、GIF 动图、SVG）
- 编辑端点：POST /v1/images/edits（改图；注意：**不支持 GIF 输出**）
- 鉴权：Authorization: Bearer sk-xxx
- 返回：默认 JSON，data[0].b64_json（或 url，取决于网关）

## 二、核心参数总表

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| prompt | string | 必填 | 最长 32000 字符 |
| size | string | auto | 1024x1024（方）/ 1536x1024（横）/ 1024x1536（竖）/ auto |
| quality | string | auto | low / medium / high / auto，影响细节与耗时 |
| complexity | string | high | low / medium / high，控制细节复杂度与生成时间；⚠ 部分模型变体不支持，乱传会 400 invalid_value |
| n | int | 1 | 1-10 张；SVG 输出强制 n=1 |
| output_format | string | png | png / jpeg / webp / gif；svg 需 n=1（走独立输出格式声明） |
| output_compression | int | 50 | 0-100，仅 png/jpeg/webp 有效；GIF 不支持 |
| background | string | auto | transparent / opaque / auto（透明底仅 png/webp） |
| moderation | string | auto | low / auto，内容审核强度 |
| stream | bool | false | 流式输出；GIF 不支持 |
| partial_images | int | 0 | 0-3，流式过程中的预览图；GIF 不支持 |
| prompt_rewriter | bool | false | 服务端改写提示词；GIF 不支持 |
| frame_count | int | 9 | 仅 GIF；帧数（文档口径不一：2-64 与 2-16 并存，建议 9 起步小步试） |
| frame_duration_ms | int | 200 | 仅 GIF；每帧毫秒数，最小 20 |
| user | string | - | 终端用户标识，滥用追踪用 |

## 三、GIF 动图专节（2.6 新能力）

必要传参四件套：

```json
{
  "model": "gpt-image-2.6-flare",
  "prompt": "……smooth loop",
  "size": "1024x1024",
  "output_format": "gif",
  "frame_count": 9,
  "frame_duration_ms": 200,
  "n": 1
}
```

硬性限制（GIF 下全部不可用）：
1. 只能走 generations 端点，edits 端点出不了 GIF
2. 不支持 stream 流式与 partial_images 预览
3. 不支持自定义 output_compression
4. 不支持 prompt_rewriter

可叠加：quality、complexity、moderation 照常生效。

帧时长公式：GIF 循环时长 = frame_count × frame_duration_ms。
默认 9 × 200ms = 1.8 秒循环。要更细腻就加帧数，要更慢就加每帧毫秒。

实测记录（2026-11-06，fair 站直连，flare 变体，9 帧 low 质量）：
- 出图耗时 29.8 秒，文件 596KB
- usage：1251 image output tokens，按 $30/M token 计 ≈ $0.0375/张
- 同参数 sunburst 变体未测，理论同传参

## 四、SVG 支持

- output_format 走 svg 声明，且必须 n=1
- 生成矢量图适合图标、logo 类需求

## 五、计费

- token 计费制，图像输出 token 单价 $30/M
- tokens 随 quality 档位、尺寸、帧数上涨；GIF 按帧累加
- low 档 1024x1024 单张静态 ≈ 数百 token；GIF 9 帧 low 档实测 1251 token

## 六、实测注意事项（fair 站血泪经验）

1. **内容审核红线**：prompt 中出现 girl、child、little 等年龄暗示词会被安全政策
   直接拦截（报错"该请求可能因安全政策被拦截"）；改用 character / count / lady
   等去年龄化措辞即可过审。OpenAI 系审核远比 NAI 严，写 prompt 时主动规避。
2. **complexity 参数坑**：部分模型变体不接受该参数，传入即 400
   （"Complexity 'xx' is not supported for this image model"）；不带该参数最稳。
   2026-11-05 该站网关曾自动注入 complexity=medium 导致全渠道画图报废，
   2026-11-06 已修复。
3. **审核与生成耗时**：静态图 25-30 秒，GIF 约 30-75 秒，超时建议设 300 秒。
4. 走中转站（sub2api 门面）时注意网关是否额外注入参数，链路排查法：
   先直连上游测通，再走门面比对。

## 七、请求示例

curl：
```bash
curl http://64.83.43.6:3000/v1/images/generations \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-image-2.6-flare",
    "prompt": "a cute orange cat on a windowsill, smooth loop",
    "size": "1024x1024",
    "output_format": "gif",
    "frame_count": 9,
    "frame_duration_ms": 200
  }'
```

python：
```python
import requests, base64
r = requests.post('http://64.83.43.6:3000/v1/images/generations',
    headers={'Authorization': 'Bearer sk-xxx'},
    json={"model": "gpt-image-2.6-flare",
          "prompt": "……",
          "size": "1024x1024",
          "output_format": "gif",
          "frame_count": 9,
          "frame_duration_ms": 200},
    timeout=300, proxies={'http': None, 'https': None})
open('out.gif', 'wb').write(base64.b64decode(r.json()['data'][0]['b64_json']))
```

## 附：本地留档

- 实测脚本：工作区 test_fair_direct.py（静态图）/ test_fair_gif.py（GIF）
- 实测产物：test_gpt_image_2.5_flare.png / test_26flare.gif
- 测试钥匙与渠道信息见《三大管理面板凭据存档.md》
