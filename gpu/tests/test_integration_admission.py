"""No-network checks for the public launch-spec/runtime boundary."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reviewed_launcher',ROOT/'gpu/run_daytona.py')
rd=importlib.util.module_from_spec(spec);sys.modules[spec.name]=rd;spec.loader.exec_module(rd)

class AdmissionTests(unittest.TestCase):
    def test_compile_spec_has_no_placeholder_or_smoke_execution(self):
        data=json.loads((ROOT/'gpu/specs/compile-duck-cuda.json').read_text())
        self.assertEqual([job['name'] for job in data['jobs']],['gpu-info','build'])
        for job in data['jobs']:
            self.assertFalse(job['continue_on_error'])
            self.assertIn('pipefail',job['command'])
            self.assertNotIn('PLACEHOLDER',job['command'])
        self.assertIn('./build_remote.sh',data['jobs'][1]['command'])

    def test_unsupported_sdk_rejected_before_client_import(self):
        with patch.dict(rd.os.environ,{'DAYTONA_API_KEY':'offline-test-placeholder'}), \
             patch.object(rd.importlib.metadata,'version',return_value='0.198.0'), \
             patch.dict(sys.modules,{'daytona':None}):
            with self.assertRaisesRegex(rd.LauncherError,'unsupported Daytona runtime'):
                rd.DaytonaProvider()

if __name__=='__main__':unittest.main()
