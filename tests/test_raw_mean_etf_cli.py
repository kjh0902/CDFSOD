"""Exercise the experiment CLI without a dataset, checkpoint, or GPU."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')
if os.name == 'nt' and Path('C:/Program Files/Git/bin/bash.exe').is_file():
    BASH = 'C:/Program Files/Git/bin/bash.exe'


@unittest.skipUnless(BASH, 'Requires Bash')
class RawMeanCLIIntegrationTests(unittest.TestCase):
    def run_cli(self, *args, root=ROOT, env=None):
        return subprocess.run([BASH, (root / 'run_cdfsod.sh').as_posix(), *args],
                              cwd=root, env=env, capture_output=True,
                              text=True, encoding='utf-8', timeout=20)

    def test_all_dataset_shot_defaults_and_invalid_weights(self):
        for dataset in ['ArTaxOr', 'Clipart1k', 'DIOR', 'FISH', 'NEU-DET', 'UODD']:
            for shot in ['1', '5', '10']:
                result = self.run_cli('--dataset', dataset, '--shot', shot, '--dry-run')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('from config (default: 1.0)', result.stdout)
        for value in ['', '-0.1', 'nan', 'inf', 'bad', '1; exit 0']:
            result = self.run_cli('--dataset', 'NEU-DET', '--shot', '1',
                                  '--etf-loss-weight', value, '--dry-run')
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        result = self.run_cli('--dataset', 'NEU-DET', '--shot', '1', '--etf-loss-weight')
        self.assertEqual(result.returncode, 2)

    def test_weight_override_reaches_training_and_resume_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix='acl-hed-etf-cli-') as directory:
            root = Path(directory)
            (root / 'run_cdfsod.sh').write_text(
                (ROOT / 'run_cdfsod.sh').read_text(encoding='utf-8'),
                encoding='utf-8', newline='\n')
            configs = root / 'configs_cdfsod/final_configs_bs4'
            configs.mkdir(parents=True)
            (configs / 'grounding_dino_swin-b_finetune_NEU-DET_1shot.py').touch()
            tools = root / 'tools'
            tools.mkdir()
            train = tools / 'dist_train.sh'
            train.write_text('''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$@" > "${TRAIN_CAPTURE}"
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --work-dir ]]; then
    touch "$2/best_coco_bbox_mAP_iter_1.pth"
    break
  fi
  shift
done
''', encoding='utf-8', newline='\n')
            test = tools / 'dist_test.sh'
            test.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8', newline='\n')
            train.chmod(0o755)
            test.chmod(0o755)
            capture = root / 'train_args.txt'
            env = dict(os.environ, TRAIN_CAPTURE=capture.as_posix())
            for option, weight in [('--etf-loss-weight', '0'),
                                   ('--etf-loss-weight', '0.25'),
                                   ('--raw-mean-etf-loss-weight', '1e-2')]:
                result = self.run_cli('--dataset', 'NEU-DET', '--shot', '1',
                                      option, weight, '--resume', root=root, env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = capture.read_text(encoding='utf-8').splitlines()
                index = args.index('--cfg-options')
                self.assertEqual(args[index + 1], f'model.raw_mean_etf_loss_weight={weight}')
                self.assertIn('--resume', args)
            result = self.run_cli('--dataset', 'NEU-DET', '--shot', '1', root=root, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn('--cfg-options', capture.read_text(encoding='utf-8').splitlines())


if __name__ == '__main__':
    unittest.main()
