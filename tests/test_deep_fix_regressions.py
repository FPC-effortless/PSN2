import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
class InputBoundaryRegression(unittest.TestCase):
 def test_ids_are_process_stable(self):
  path=ROOT/'psn2/token_ids.py'
  code="import importlib.util,json;s=importlib.util.spec_from_file_location('ids',"+repr(str(path))+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(json.dumps([m.stable_token_id(x,4096) for x in ['abc','def','ghi']]))"
  values=[subprocess.check_output([sys.executable,'-c',code],env={**os.environ,'PYTHONHASHSEED':v},text=True) for v in ['1','2']]
  self.assertEqual(values[0],values[1])
 def test_bbh_test_is_not_a_training_path(self):
  s=(ROOT/'train.py').read_text();self.assertNotIn('bbh_test',s);self.assertIn('BBHDataset(bbh_train',s)
 def test_legacy_language_adapter_rejects_before_input(self):
  tree=ast.parse((ROOT/'psn2/datasets/wikitext_dataset.py').read_text())
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WikitextDataset')
  fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
  fn.decorator_list=[]
  scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original-guard>','exec'),scope)
  with self.assertRaises(NotImplementedError):scope['__init__'](None,'nonexistent.jsonl')
 def test_checkpoint_rejects_unknown_token_identity(self):
  tree=ast.parse((ROOT/'psn2/core.py').read_text())
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PSN2System')
  fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='load_state_dict_full')
  scope={'TOKEN_ID_VERSION':'blake2b-psn2-id-v2'}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original-load-guard>','exec'),scope)
  with self.assertRaises(ValueError):scope['load_state_dict_full'](None,{})
 def test_real_dataset_guard_if_torch_available(self):
  if importlib.util.find_spec('torch') is None:self.skipTest('PyTorch runtime unavailable')
  sys.path.insert(0,str(ROOT))
  from psn2.datasets.wikitext_dataset import WikitextDataset
  with self.assertRaises(NotImplementedError):WikitextDataset('nonexistent.jsonl')
if __name__=='__main__':unittest.main()
