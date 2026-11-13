import unittest
import tempfile
import json
from pathlib import Path
from scripts.execute_notebooks import execute


class NotebookTests(unittest.TestCase):
    def test_notebook_execution_persists_outputs_and_rejects_cell_errors(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'toy.ipynb'
            path.write_text(json.dumps({'cells':[{'cell_type':'code','source':['x=2\nprint(x+3)'],'metadata':{},'outputs':[],'execution_count':None}], 'metadata':{},'nbformat':4,'nbformat_minor':5}))
            execute(path)
            d=json.loads(path.read_text()); self.assertEqual(d['cells'][0]['outputs'][0]['text'],['5\n'])
            d['cells'][0]['source']=['raise ValueError("bad")'];path.write_text(json.dumps(d))
            with self.assertRaises(ValueError): execute(path)
