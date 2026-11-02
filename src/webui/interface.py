"""Gradio demo for GlyphAnchor generation, layout editing, and layout export."""
import asyncio
import json
import os
from pathlib import Path
import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import tempfile

from ..inputs import parse_layout_response
from .layout_editor import editor_layout
from .preview import render_layout_preview

GPU_DEVICE = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
_WORKER = None


def get_worker():
    global _WORKER
    from .worker import initialize_worker
    if _WORKER is None:
        _WORKER = ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker, initargs=(GPU_DEVICE,))
    return _WORKER


@atexit.register
def shutdown_worker():
    if _WORKER is not None:
        _WORKER.shutdown(wait=False, cancel_futures=True)


ROOT = Path(__file__).resolve().parents[2]
MODELS = ['GlyphAnchor-Z-Image', 'GlyphAnchor-FireRed-Image-Edit-1.1',
          'GlyphAnchor-Qwen-Image-Edit-2511']
EXAMPLES = {'Academic slide': 'example_layout_1.json',
            'Journaling guide (text-to-image)': 'example_layout_2.json',
            'Journaling guide (editing)': 'example_layout_3.json',
            'Relativity with My Melody (text-to-image)': 'example_layout_4.json',
            'Relativity with My Melody (editing)': 'example_layout_5.json'}


def example_inputs(name):
    path = ROOT / 'examples' / EXAMPLES[name]
    data = json.loads(path.read_text())
    refs = [str(path.parent / p) for p in data.get('images', [])]
    return (data['prompt'], json.dumps(editor_layout(data.get('layout') or []), ensure_ascii=False),
            data.get('width', 1152), data.get('height', 2048), *(refs + [None] * (4-len(refs))))


def make_payload(prompt, layout, width, height, images):
    if not prompt or not prompt.strip():
        raise ValueError('Enter a prompt first.')
    if any(int(v) != v or v <= 0 or v % 32 for v in (width, height)):
        raise ValueError('Width and height must be positive multiples of 32.')
    return {'prompt': prompt.strip(), 'layout': editor_layout(parse_layout_response(layout)),
            'width': int(width), 'height': int(height),
            'images': [str(Path(p).resolve()) for p in images if p]}


def export_layout(model, prompt, layout, width, height, *images):
    data = make_payload(prompt, layout, width, height, [] if model == MODELS[0] else images)
    folder = Path(tempfile.mkdtemp(prefix='glyphanchor-showcase-'))
    path = folder / 'example_layout.json'
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    return str(path)


def plan(prompt, width, height, backend, url, model, *images, options=None, api_key=''):
    from ..vlm_backend import call_vlm_for_layout
    if not prompt.strip():
        raise ValueError('Enter a prompt.')
    content, _ = asyncio.run(call_vlm_for_layout(
        prompt, [p for p in images if p], (int(width), int(height)),
        backend=backend, base_url=url, model=model,
        api_key=api_key.strip() or 'EMPTY', options=options))
    return json.dumps(editor_layout(parse_layout_response(content)), ensure_ascii=False)


def rewrite_prompt(prompt, mode, backend, url, model, *images, options=None, api_key=''):
    from ..vlm_backend import call_vlm_for_prompt_rewrite
    if not prompt.strip():
        raise ValueError('Enter a prompt.')
    rewritten, elapsed = asyncio.run(call_vlm_for_prompt_rewrite(
        prompt, [p for p in images if p], backend=backend, mode=mode,
        base_url=url, model=model, api_key=api_key.strip() or 'EMPTY', options=options))
    if not rewritten:
        raise ValueError('VLM returned an empty prompt.')
    return rewritten, f'Rewrite completed in {elapsed:.1f}s'


def sync_layout_js(index):
    return """(...args) => {
        document.getElementById('glyph-text-editor')?.blur();
        window._glyphEditor?.sync();
        const node = document.querySelector('#layout-json-store textarea, #layout-json-store input');
        if (node) args[INDEX] = node.value;
        return args;
    }""".replace('INDEX', str(index))


def submit_generation(model, source, base_model, data, seed, steps, guidance):
    from .worker import generate_on_worker
    from ..generation import resolve_model_type
    model_type = resolve_model_type(model)
    source = source.strip()
    if not source:
        raise ValueError('Enter a checkpoint directory, checkpoint root, or Hugging Face repository ID.')
    if Path(source).expanduser().is_dir():
        local_path = Path(source).expanduser().resolve()
        checkpoint = str(local_path if (local_path / 'config.json').is_file() else local_path / model)
        subfolder = None
    else:
        checkpoint, subfolder = source, model
    output = Path(tempfile.mkdtemp(prefix='glyphanchor-generation-')) / 'output.png'
    return get_worker().submit(generate_on_worker, checkpoint, subfolder,
                                   (base_model or '').strip(), data, seed, steps, guidance, str(output),
                                   model_type=model_type)


def generate(model, source, base_model, prompt, layout, width, height, seed, steps, guidance, *images):
    if model == MODELS[0] and any(images):
        raise ValueError('Z-Image supports text-to-image only. For an editing case with reference images, select FireRed or Qwen.')
    data = make_payload(prompt, layout, width, height, images)
    return submit_generation(model, source, base_model, data, seed, steps, guidance).result()


def build_app(config=None):
    import gradio as gr
    from .layout_editor import CANVAS_HTML, CANVAS_JS
    config = config or {}
    server_key = os.environ.get('VLM_API_KEY', '').strip()
    if server_key == 'EMPTY':
        server_key = ''
    server_url = config.get('url', os.getenv('VLM_URL', ''))
    default_example = next(iter(EXAMPLES))
    initial_prompt, initial_layout, initial_width, initial_height, *initial_refs = example_inputs(default_example)

    def guarded(fn):
        def call(*args):
            try:
                return fn(*args)
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from None
            except Exception:
                import traceback
                traceback.print_exc()
                # Do not expose service URLs or credentials in browser errors.
                raise gr.Error('Request failed. See the terminal for details.') from None
        return call

    async def load_example(name):
        return example_inputs(name)

    def plan_ui(prompt, width, height, backend, model, options, ref1, ref2, ref3, ref4):
        images = (ref1, ref2, ref3, ref4)
        try:
            result = plan(prompt, width, height, backend, server_url, model, *images, options=options, api_key=server_key)
            return result, f'Layout ready: {len(json.loads(result))} blocks'
        except Exception as exc:
            raise gr.Error(f'Layout planning failed: {exc}') from None

    def rewrite_ui(prompt, mode, backend, model, options, ref1, ref2, ref3, ref4):
        images = (ref1, ref2, ref3, ref4)
        return guarded(lambda: rewrite_prompt(prompt, mode, backend, server_url, model, *images, options=options, api_key=server_key))()

    with gr.Blocks(title='GlyphAnchor') as demo:
        gr.Markdown('# GlyphAnchor\nPlan a layout, edit the text boxes, and generate an image.')
        with gr.Row():
            with gr.Column(scale=4, min_width=320):
                selected = gr.Dropdown(MODELS, value=MODELS[0], label='Model')
                source = gr.Textbox(value='innerway/GlyphAnchor', label='Checkpoint directory or root / Hugging Face repository')
                base_model = gr.Textbox(value='', label='Base model (optional local path or Hub ID)',
                                        placeholder='Leave empty to load the default from Hugging Face')
                prompt = gr.Textbox(value=initial_prompt, label='Prompt', lines=7)
                with gr.Row():
                    rewrite_btn = gr.Button('Rewrite Prompt')
                    rewrite_mode = gr.Radio(['default', 'enrich'], value='default', label='Rewrite mode')
                rewrite_status = gr.Textbox(label='Rewrite status', interactive=False)
                with gr.Row():
                    refs = [gr.Image(value=initial_refs[i], type='filepath', label=f'Reference {i+1}') for i in range(2)]
                with gr.Accordion('More reference images', open=False):
                    refs += [gr.Image(value=initial_refs[i], type='filepath', label=f'Reference {i+1}') for i in range(2,4)]
                gr.Markdown('Reference images are used by FireRed and Qwen. Z-Image generates from text.')
                with gr.Row():
                    width = gr.Slider(512, 2048, value=initial_width, step=32, label='Width')
                    height = gr.Slider(512, 2048, value=initial_height, step=32, label='Height')
                with gr.Accordion('Generation settings', open=False):
                    seed = gr.Number(value=42, precision=0, label='Seed')
                    steps = gr.Slider(1, 100, value=50, step=1, label='Steps')
                    guidance = gr.Number(value=4.0, label='Guidance scale')
                with gr.Accordion('VLM settings', open=False):
                    backend = gr.Dropdown(['openai', 'google'], value=config.get('backend', 'openai'), label='API protocol')
                    gr.Textbox(value=server_url, label='VLM base URL / endpoint', interactive=False)
                    vlm_model = gr.Textbox(value=config.get('model', os.getenv('VLM_MODEL','')), label='VLM model')
                    vlm_options = gr.Textbox(value=json.dumps(config.get('options', {}), ensure_ascii=False), label='Protocol / request options (JSON)', lines=3)
            with gr.Column(scale=7, min_width=520):
                plan_btn = gr.Button('Plan Layout')
                layout_status = gr.Textbox(label='Layout status', interactive=False)
                layout = gr.Textbox(value=initial_layout, elem_id='layout-json-store', label='Layout JSON')
                with gr.Row():
                    with gr.Column():
                        gr.HTML(CANVAS_HTML)
                    with gr.Column():
                        refresh_preview = gr.Button('Refresh Glyph Preview')
                        preview = gr.Image(label='Glyph preview (selected model)', type='pil', interactive=False)
                        gr.Markdown('Actual rendered glyph patches and final text boxes for the selected model.')
                generate_btn = gr.Button('Generate', variant='primary')
                output = gr.Image(label='Generated image', type='filepath')
                save_btn = gr.Button('Save Layout')
                layout_download = gr.File(label='Layout JSON')
        example_name = gr.Textbox(visible=False, label='Example')
        gr.Examples(examples=[[name] for name in EXAMPLES], inputs=[example_name],
                    outputs=[prompt,layout,width,height,*refs], fn=load_example,
                    cache_examples=False, run_on_click=True, label='Examples')
        preview_inputs = [selected, layout, width, height]
        for component in [selected, layout, width, height]:
            component.change(guarded(render_layout_preview), preview_inputs, preview,
                             concurrency_limit=1, concurrency_id='preview', trigger_mode='always_last')
        refresh_preview.click(guarded(render_layout_preview), preview_inputs, preview,
                              concurrency_limit=1, concurrency_id='preview', js=sync_layout_js(1))
        demo.load(guarded(render_layout_preview), preview_inputs, preview)
        selected.change(lambda model: '', [selected], base_model, queue=False)
        plan_btn.click(plan_ui, [prompt,width,height,backend,vlm_model,vlm_options,*refs],
                       [layout,layout_status], show_progress='full', show_progress_on=[layout_status])
        rewrite_btn.click(rewrite_ui, [prompt,rewrite_mode,backend,vlm_model,vlm_options,*refs],
                          [prompt,rewrite_status], show_progress='full', show_progress_on=[rewrite_status])
        generate_btn.click(guarded(generate), [selected,source,base_model,prompt,layout,width,height,seed,steps,guidance,*refs], output,
                           concurrency_limit=1, concurrency_id='gpu', js=sync_layout_js(4))
        save_btn.click(guarded(export_layout), [selected,prompt,layout,width,height,*refs], layout_download, js=sync_layout_js(2))
        aspect_js = '(w,h) => {window._glyphEditor?.setAspect(w,h);}'
        width.change(None, [width,height], None, js=aspect_js)
        height.change(None, [width,height], None, js=aspect_js)
        demo.load(None, None, None, js=CANVAS_JS.replace(
            "setAspect(1152, 2048);", f"setAspect({initial_width}, {initial_height});"))
    return demo
