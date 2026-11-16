import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

class ApiTests(unittest.TestCase):
 def setUp(self):
  self.env=patch.dict(os.environ,{},clear=True);self.env.start()
  sdk=ModuleType("openai")
  async def create(**kwargs):
   async def stream():
    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="A small step can help."))])
   return stream()
  sdk.AsyncOpenAI=lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  rag=ModuleType("rag");rag.retrieve_context=lambda query:""
  self.modules=patch.dict(sys.modules,{"openai":sdk,"rag":rag});self.modules.start()
  spec=importlib.util.spec_from_file_location("support_api",Path(__file__).resolve().parents[1]/"main.py")
  self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
  self.client=TestClient(self.module.app)
 def tearDown(self):self.modules.stop();self.env.stop()
 def test_validation_and_missing_provider(self):
  self.assertEqual(self.client.get('/').status_code,200)
  self.assertEqual(self.client.post('/chat',json={'message':12}).status_code,422)
  self.assertEqual(self.client.post('/chat',json={'message':'x'*4001}).status_code,422)
  self.assertEqual(self.client.post('/chat',json={'message':'hello'}).status_code,503)
 def test_history_is_attached_to_server_cookie_not_body_id(self):
  os.environ['GROQ_API_KEY']='test-only'
  first=self.client.post('/chat',json={'message':'hello','session_id':'attacker-chosen'})
  self.assertEqual(first.status_code,200)
  cookie=self.client.cookies.get('support_session')
  self.assertNotEqual(cookie,'attacker-chosen')
  self.assertIn('HttpOnly',first.headers['set-cookie'])
  self.client.post('/chat',json={'message':'another message','session_id':'different'})
  self.assertEqual(len(self.module.sessions),1)
  self.assertEqual(len(self.module.sessions[cookie]['history']),4)
  self.assertEqual(self.client.post('/chat',json={'message':'hello'},headers={'origin':'https://untrusted.example'}).status_code,403)
if __name__=='__main__':unittest.main()
