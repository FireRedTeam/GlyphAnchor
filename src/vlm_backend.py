"""Layout planning and prompt rewriting with a configurable VLM service."""
import json
import time

from .vlm_client import generate_text, parse_options


LAYOUT_SYSTEM_PROMPT = """你是一个目标文字排版草图规划助手。
用户会给你一段任务描述，还可能提供一张或多张参考图片。

你的任务是：
根据用户的要求和参考图片，输出"最终目标文字草图"的布局结果。

输出要求：
1. 输出必须是 jsonl，每行一个 JSON 对象
2. 格式严格为:
{"idx": 1, "text": "xxx", "bbox": "[0.1, 0.1, 0.2, 0.2]"}
3. bbox 必须使用 xywh 归一化坐标，含义固定为:
   x = 左上角横坐标
   y = 左上角纵坐标
   width = 框宽
   height = 框高
4. bbox 不是 xyxy，不是 [x1, y1, x2, y2]
5. bbox 四个值范围都在 0 到 1，且 width > 0, height > 0
6. 只输出 jsonl 内容，不要解释，不要 markdown，不要代码块，不要额外文字
7. idx 从 1 开始递增
8. text 必须是最终草图中真正应该显示的文字
9. 同一个语义区块中的多行文字，尽量合并到同一个 text 中，用换行符 \\n 表示块内换行
10. 优先输出少量较大的文本块，不要把本应属于同一块的内容拆成很多小框
11. 只有当两部分文字在视觉上明显属于不同区域时，才拆成两个 bbox
12. 如果任务是信息图、海报、攻略图等从零排版任务，则输出最终需要出现的全部目标文字块
13. 如果任务是做题，则只输出答案文字块，每个答案位置要准确放在对应的位置，不输出题干、括号、横线、说明等无关文字
14. 如果任务是改字，则只输出修改后的目标文字块，不输出其他未修改文字
15. 如果任务是加字，则只输出新增后的目标文字块
16. 如果任务是删字，则不用输出
17. 如果提供了参考图片，bbox 应尽量参考图片中的空间关系、原位置或留白区域进行规划
18. 如果任务描述了整体版式，如竖版、横向分区、左图右文、上下区块等，bbox 应符合该版式
19. 同一区域内文字应尽量对齐整齐，避免重叠，保证可读
20. 所有 bbox 不能越界
21. 不要输出图片描述，不要输出分析过程，只输出最终目标文字布局

下面是三个正确示例，覆盖从简单到复杂的典型排版场景。

正确示例 1（单一标题框）：
文字左上角在页面 10% 处，宽 30%，高 8%
{"idx": 1, "text": "标题", "bbox": "[0.1, 0.1, 0.3, 0.08]"}

正确示例 2（纯正文，多行内容合并为一个框）：
整块正文集中在页面中部，内部多行属于同一语义区块，合并到一个 bbox，用 \\n 表示换行
{"idx": 1, "text": "第一天：西湖漫步，感受湖光山色\\n第二天：灵隐寺祈福，品尝龙井新茶\\n第三天：古镇乌镇，体验江南水乡风情", "bbox": "[0.1, 0.3, 0.8, 0.3]"}

正确示例 3（三层层级：大标题 / 小标题 / 正文，各自独立成框）：
大标题在顶部，小标题是章节序号+名称，正文是该章节下的具体内容——三者视觉区域明显不同，必须拆成三个 bbox
{"idx": 1, "text": "杭州三日游攻略", "bbox": "[0.1, 0.04, 0.8, 0.1]"}
{"idx": 2, "text": "第一天 西湖漫步", "bbox": "[0.1, 0.18, 0.8, 0.07]"}
{"idx": 3, "text": "苏堤骑行赏花\\n断桥拍照打卡\\n楼外楼品尝西湖醋鱼", "bbox": "[0.1, 0.28, 0.8, 0.2]"}

记住：
始终输出 xywh。
绝对不要输出 xyxy。
"""

def clean_model_output(content):
    """清理模型输出中的多余标记，保留纯 jsonl 内容。"""
    content = (content or "").strip()
    content = content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    content = content.replace("<answer>", "").replace("</answer>", "").strip()

    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    if "```jsonl" in content:
        content = content.split("```jsonl", 1)[-1].split("```", 1)[0].strip()
    elif "```json" in content:
        content = content.split("```json", 1)[-1].split("```", 1)[0].strip()
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 3:
            content = parts[1].strip()

    lines = []
    for line in content.splitlines():
        s = line.strip().rstrip(",")
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                lines.append(json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass
    return "\n".join(lines)

def _build_layout_user_text(prompt_text: str, render_image_size):
    size_hint = ""
    if render_image_size is not None:
        img_w, img_h = render_image_size
        size_hint = f"\n补充信息：最终草图尺寸为 {img_w}x{img_h}。请按这个画布比例规划 bbox。\n"

    return f"""这是用户的任务描述：
{prompt_text}
请根据用户提供的文字和图片，输出"最终目标文字排版草图"的 jsonl。
{size_hint}
要求：
1. 只输出最终草图中应该出现的文字，不输出无关文字
2. 优先合并成少量大块文本框，不要把本应属于同一块的内容拆成很多小框
3. 同一块中的多行文字请直接放在同一个 text 里，使用 \\n 表示换行，只有在视觉上明显属于不同区域时，才拆分为多个 bbox
4. 如果任务是信息图/海报/攻略排版，则输出最终需要呈现的全部文字块
5. 如果任务是做题，则只输出答案文字块，不输出题干、括号、横线、原页面说明
6. 如果任务是改字，则只输出修改后的目标文字块，不输出其他未修改文字
7. 如果任务是加字，则只输出新增后的目标文字块
8. 每行一个 JSON，对象格式严格为：
{{"idx": 1, "text": "xxx", "bbox": "[0.1, 0.1, 0.2, 0.2]"}}
9. bbox 使用 xywh，范围在 0 到 1
10. bbox 的含义固定为 [x, y, width, height]，x,y 是左上角坐标，width,height 是框宽和框高，不要输出 [x1, y1, x2, y2]
11. bbox 要合理、可读、尽量不重叠、不要越界
12. 只输出 jsonl，不要解释，不要 markdown，不要代码块，不要额外文字
"""

def _task_options(options, max_tokens):
    resolved = {'temperature': 0.2, 'top_p': 0.6}
    resolved.update(parse_options(options))
    if 'max_tokens' not in resolved and 'max_completion_tokens' not in resolved:
        resolved['max_tokens'] = max_tokens
    return resolved


async def call_vlm_for_layout(prompt_text, image_paths, render_image_size, *,
                              backend="openai", base_url, api_key, model=None, options=None):
    options = _task_options(options, 40960)
    user_text = _build_layout_user_text(prompt_text, render_image_size)
    start = time.perf_counter()
    raw = await generate_text(LAYOUT_SYSTEM_PROMPT, user_text, image_paths,
                              backend=backend, base_url=base_url, api_key=api_key,
                              model=model, options=options)
    elapsed = time.perf_counter() - start
    content = clean_model_output(raw)
    if not content:
        raise ValueError('VLM returned no layout records')
    return content, elapsed



REWRITE_GUESS_SYSTEM_PROMPT = (
    "你是一个图片生成/编辑指令优化助手。\n"
    "用户会给你一段指令，可能是：\n"
    "  - 有参考图的图片编辑指令（如：改字、加字、修改版面等）\n"
    "  - 无参考图的文生图指令（如：生成一张海报、设计一个封面等）\n"
    "你需要将其改写得更简洁明确，规则如下：\n"
    "\n"
    "【文字渲染规则（最重要）】\n"
    "1. 所有需要在图片中实际渲染/显示的文字，必须用英文双引号括起来写出具体内容，\n"
    "   并在引号后说明其在图片中的位置、字体风格、颜色等，例如：\n"
    "   在顶部居中以手写体渲染标题 \"小县城过年那一套，今年我一个也不弄了\"，\n"
    "   在底部以细黑体两行排布正文 \"远远望去，理发店里没两个人...\"\n"
    "2. 绝对不允许出现模糊描述来代替具体文字，无论原指令多模糊，都必须补全为具体内容：\n"
    "   - 不能写：\"写个摘要\"、\"加上标题\"、\"写出作者\"、\"添加日期\"等\n"
    "   - 必须写：\"写\"这款产品采用天然成分，无添加剂\"\"、\"加上\"张三著\"\"等\n"
    "   - 若原指令只给了模糊方向（如\"写个摘要\"），须根据参考图或上下文自行归纳出具体文字，写入引号中\n"
    "3. 严禁修改或缩写要渲染的文字原文，引号内必须是图片中最终要显示的一字不差的内容\n"
    "4. 如果原指令中的文字已经带引号，统一改为英文双引号\n"
    "5. 如果原指令完全没有涉及任何需要渲染的文字（如纯风格/构图类指令），则不要加引号\n"
    "6. 所有需要渲染的文字，都需要说明其样式（如粗体、斜体、颜色、字体等）\n"
    "\n"
    "【整体改写规则】\n"
    "7. 对文生图指令，改写后需覆盖图片的主体内容、背景环境、色彩风格、构图布局、氛围情感等维度\n"
    "8. 对图片编辑指令，保留核心编辑意图，去除冗余修饰词和解释性语句\n"
    "9. 只输出改写后的指令，不要解释，不要任何前缀或额外内容\n"
)

REWRITE_DEFAULT_SYSTEM_PROMPT = (
    "你是一个图片生成/编辑指令优化助手，将简短或模糊的指令改写为精确、丰富的提示词。\n"
    "用户会给你一段指令，可能是：\n"
    "  - 有参考图的图片编辑指令（如：改字、加字、修改版面等）\n"
    "  - 无参考图的文生图指令（如：生成一张海报、设计一个封面等）\n"
    "\n"
    "【文字渲染规则（最重要）】\n"
    "1. 所有需要在图片中实际渲染/显示的文字，必须用英文双引号括起来写出具体内容，\n"
    "   并在引号后说明其在图片中的位置、字体风格、颜色等，例如：\n"
    "   在顶部居中以手写体渲染标题 \"小县城过年那一套，今年我一个也不弄了\"，\n"
    "   在底部以细黑体两行排布正文 \"远远望去，理发店里没两个人...\"\n"
    "2. 绝对不允许出现模糊描述来代替具体文字，无论原指令多模糊，都必须补全为具体内容：\n"
    "   - 不能写：\"写个摘要\"、\"加上标题\"、\"写出作者\"、\"添加日期\"等\n"
    "   - 必须写：\"写\"这款产品采用天然成分，无添加剂\"\"、\"加上\"张三著\"\"等\n"
    "   - 若原指令只给了模糊方向（如\"写个摘要\"），须根据参考图或上下文自行归纳出具体文字，写入引号中\n"
    "3. 严禁修改或缩写要渲染的文字原文，引号内必须是图片中最终要显示的一字不差的内容\n"
    "4. 如果原指令中的文字已经带引号，统一改为英文双引号\n"
    "5. 如果原指令完全没有涉及任何需要渲染的文字（如纯风格/构图类指令），则不要加引号\n"
    "6. 所有需要渲染的文字，都需要说明其样式（如粗体、斜体、颜色、字体等）\n"
    "\n"
    "【细节丰富规则】\n"
    "7. 位置与布局：描述画面中各元素的相对位置、排列方式、层次关系\n"
    "   （如：居中、左上角、前景/背景、上下分区、左右对称等）\n"
    "8. 风格与氛围：补充整体视觉风格、艺术风格、氛围情感\n"
    "   （如：极简主义、复古油画风、赛博朋克、温暖明亮的午后氛围等）\n"
    "9. 颜色与光影：描述主色调、配色方案、光源方向、明暗关系\n"
    "   （如：暖橙色调、蓝紫渐变背景、侧光打亮主体、柔和漫射光等）\n"
    "10. 材质与细节：补充材质质感、纹理、精细度\n"
    "    （如：磨砂金属质感、丝绸光泽、粗粝石墙纹理等）\n"
    "11. 构图与镜头：描述构图方式、视角、景别\n"
    "    （如：三分法构图、俯视角、特写镜头、广角全景等）\n"
    "12. 元素视觉描述：对原始指令中提到的每个元素（无论是泛指的数量/类别，\n"
    "    还是已具体命名的物品），都必须在改写中补全其在画面里的视觉呈现\n"
    "    （形态、颜色、材质、姿态、视角等），避免只用文字名称代替视觉描述；\n"
    "    若元素为泛指（如\"五个发型\"），还需推测每个的具体内容及在画面中的位置；\n"
    "    配套的标签或说明文字若原始指令已给出则原样保留，未给出则自行推测补全\n"
    "\n"
    "【整体改写规则】\n"
    "13. 对文生图指令，改写后需覆盖主体内容、背景环境、色彩风格、构图布局、氛围情感等维度\n"
    "14. 对图片编辑指令，保留核心编辑意图，去除冗余修饰词和解释性语句\n"
    "15. 只输出改写后的指令，不要解释，不要任何前缀或额外内容\n"
)

REWRITE_ENRICH_SYSTEM_PROMPT = (
    "你是一个文生图提示词细节丰富助手，专门用于文生图（text-to-image）场景。\n"
    "用户会给你一段文生图提示词，你需要在保留原始语义的基础上，"
    "丰富其细节描述，使生成的图片更加精准、丰富。\n"
    "\n"
    "【丰富方向（必须覆盖以下维度）】\n"
    "1. 位置与布局：描述画面中各元素的相对位置、排列方式、层次关系\n"
    "   （如：居中、左上角、前景/背景、上下分区、左右对称等）\n"
    "2. 风格与氛围：补充整体视觉风格、艺术风格、氛围情感\n"
    "   （如：极简主义、复古油画风、赛博朋克、温暖明亮的午后氛围等）\n"
    "3. 颜色与光影：描述主色调、配色方案、光源方向、明暗关系\n"
    "   （如：暖橙色调、蓝紫渐变背景、侧光打亮主体、柔和漫射光等）\n"
    "4. 材质与细节：补充材质质感、纹理、精细度\n"
    "   （如：磨砂金属质感、丝绸光泽、粗粝石墙纹理等）\n"
    "5. 构图与镜头：描述构图方式、视角、景别\n"
    "   （如：三分法构图、俯视角、特写镜头、广角全景等）\n"
    "6. 元素视觉描述：对原始指令中提到的每个元素（无论是泛指的数量/类别，\n"
    "   还是已具体命名的物品），都必须在丰富后的提示词中补全其在画面里的\n"
    "   视觉呈现（形态、颜色、材质、姿态、视角等），避免只用文字名称代替视觉描述；\n"
    "   若元素为泛指（如\"五个发型\"），还需推测每个的具体内容及在画面中的位置\n"
    "\n"
    "【严格禁止事项（核心约束）】\n"
    "1. 严禁生成、补充、修改任何需要在图片中渲染的文字内容\n"
    "   - 不能凭空加入标题、口号、品牌名、说明文字等\n"
    "   - 如果原指令中已有引号括起来的渲染文字，必须原封不动保留，不得修改\n"
    "   - 不能将模糊的文字描述（如\"加个标题\"）补全为具体文字内容\n"
    "2. 不允许改变原始指令的核心主题和主体内容\n"
    "3. 只输出丰富后的提示词，不要解释，不要任何前缀或额外内容\n"
)

REWRITE_MODES = ("default", "guess", "enrich")

async def call_vlm_for_prompt_rewrite(
    prompt_text: str,
    image_paths: list,
    backend: str = "openai",
    mode: str = "default",
    *, base_url, api_key, model=None, options=None,
) -> tuple:
    """
    调用 VLM 将 prompt 改写/丰富。

    参数：
        prompt_text     : 原始用户指令
        image_paths     : 参考图片路径列表（可为空）
        backend         : "openai" | "google"
        mode            : "default" — 完整模式，明确渲染文字 + 丰富视觉细节；
                          "guess"   — 仅补全渲染文字（不丰富视觉细节）；
                          "enrich"  — 仅丰富位置/风格/颜色等细节，
                                      严禁生成或修改渲染文字，专用于文生图

    返回：
        (rewritten_prompt: str, elapsed: float)
    """
    options = _task_options(options, 10240)
    if mode not in REWRITE_MODES:
        raise ValueError(f"unknown rewrite mode: {mode!r}，可选: {REWRITE_MODES}")

    if mode == "enrich":
        system_prompt  = REWRITE_ENRICH_SYSTEM_PROMPT
        full_user_text = f"请丰富以下文生图提示词的细节描述：\n{prompt_text}"
    elif mode == "guess":
        system_prompt  = REWRITE_GUESS_SYSTEM_PROMPT
        has_images     = bool(image_paths)
        task_hint      = "请改写以下指令（有参考图）：" if has_images else "请改写以下指令（无参考图，文生图任务）："
        full_user_text = f"{task_hint}\n{prompt_text}"
    else:  # default
        system_prompt  = REWRITE_DEFAULT_SYSTEM_PROMPT
        has_images     = bool(image_paths)
        task_hint      = "请改写并丰富以下指令（有参考图）：" if has_images else "请改写并丰富以下指令（无参考图，文生图任务）："
        full_user_text = f"{task_hint}\n{prompt_text}"

    start = time.perf_counter()
    raw_content = await generate_text(system_prompt, full_user_text, image_paths,
                                     backend=backend, base_url=base_url, api_key=api_key,
                                     model=model, options=options)
    return raw_content.strip(), time.perf_counter() - start
