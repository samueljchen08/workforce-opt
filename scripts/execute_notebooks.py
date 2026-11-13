"""Execute this project's plain-Python notebook cells without a Jupyter server.

Run from repository root. This intentionally supports only code used in the
included notebooks (no IPython magics or rich-display capture). For full rich
output, open the notebooks in Jupyter instead.
"""
from pathlib import Path
import contextlib
import io
import json


def execute(path):
    path=Path(path); notebook=json.loads(path.read_text())
    namespace={'__name__':'__main__'}; count=0
    for cell in notebook['cells']:
        if cell['cell_type']!='code': continue
        count+=1; stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(''.join(cell['source']),str(path)+f':cell{count}','exec'),namespace)
        cell['execution_count']=count
        text=stream.getvalue()
        cell['outputs']=[dict(output_type='stream',name='stdout',text=text.splitlines(keepends=True))] if text else []
    path.write_text(json.dumps(notebook,indent=1)+'\n')
    print(f'Executed {path}: {count} code cells')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('notebooks',nargs='*')
    args=p.parse_args()
    for path in args.notebooks or ['notebooks/01_data.ipynb','notebooks/02_results.ipynb']: execute(path)
