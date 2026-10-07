import sys, os, types, json

os.environ['LLM_BACKEND'] = 'local'
sys.path.insert(0, '.')

# Mock gradio so we can import app without installing it
gr_mock = types.ModuleType('gradio')
sys.modules['gradio'] = gr_mock
class M:
    def __init__(self,*a,**kw): pass
    def __enter__(self): return self
    def __exit__(self,*a): pass
    def __call__(self,*a,**kw): return M()
    def click(self,*a,**kw): pass
for attr in ['Blocks','Row','Column','Markdown','Audio','Dropdown',
             'Checkbox','Textbox','Button','Tabs','Tab','File','themes']:
    setattr(gr_mock, attr, M)
gr_mock.themes = types.SimpleNamespace(Soft=M)

from app import _fmt_flags, _default_glossary
print('OK  app.py imports')
print(f'OK  Glossary: {len(_default_glossary().splitlines())} terms')
print(f'OK  Flags (empty): {_fmt_flags([])}')

flags = [{'stage':'D','type':'unsupported_task','item':'Deploy by Friday','reason':'not in transcript'}]
out = _fmt_flags(flags)
print(f'OK  Flags (1 item): {repr(out[:80])}')

fs = json.load(open('stageA/stageA_fewshot.json', encoding='utf-8'))
print(f'OK  Stage A few-shot: {len(fs)} AMI examples')

n_eval = sum(1 for _ in open('stageA/stageA_eval.jsonl', encoding='utf-8'))
print(f'OK  Eval set: {n_eval} meetings')

print()
print('All checks passed! Ready to run: python app.py')
