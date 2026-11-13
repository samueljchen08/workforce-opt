import unittest
import tempfile
import json
from pathlib import Path
import pandas as pd
from src.artifacts import build_overview


class ArtifactTests(unittest.TestCase):
    def test_overview_preserves_coverage_and_links_to_actual_scenario_files(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'reproduction'; folder.mkdir()
            (folder/'metadata.json').write_text(json.dumps({'selected_k':50,'selected_z':0,'test_weeks':34,'audit':{'D5_usable':11068}}))
            (folder/'RESULTS.md').write_text('Real results')
            pd.DataFrame([dict(method='M2 <test>',mean_hours=72,backlog_total=20,mean_coverage=.99)]).to_csv(folder/'comparison.csv',index=False)
            target=build_overview(td); text=target.read_text()
            self.assertIn('M2 &lt;test&gt;',text)
            self.assertIn('99.00%',text)
            self.assertIn('reproduction/RESULTS.md',text)
            self.assertNotIn('strict/comparison.csv',text)
