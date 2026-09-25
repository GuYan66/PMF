"""Run: python -m unittest discover -s tests -p test_pmf.py -v"""
import csv
import json
from contextlib import contextmanager
import importlib
import io
import logging
import pickle
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from easydict import EasyDict

from MMSA.config import get_config_regression
from MMSA.models import AMIO
from MMSA.models.singleTask.PMF import PMF
from MMSA.models.subNets.PolMagHead import polarity_from_intensity
from MMSA.trains import ATIO


@contextmanager
def experiment_directory():
    # Windows cannot remove a directory containing open log files.
    with tempfile.TemporaryDirectory(prefix='pmf-tests-') as tmp:
        try:
            yield tmp
        finally:
            logger = logging.getLogger('MMSA')
            for handler in list(logger.handlers):
                if getattr(handler, '_mmsa_owned', False):
                    logger.removeHandler(handler)
                    handler.close()


class PMFTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def args(self, dims=(2, 1, 1)):
        return EasyDict(polmag=True, feature_dims=dims, dropout=0.0)

    def test_padding_excluded_but_missing_zero_retained(self):
        text = torch.tensor([[[2., 4.], [0., 0.], [999., 999.]]])
        audio = torch.tensor([[[6.], [0.], [999.]]])
        vision = torch.tensor([[[8.], [0.], [999.]]])
        mask = torch.tensor([[1., 1., 0.]])
        pooled = PMF.pooled_features(text, audio, vision, mask)
        torch.testing.assert_close(pooled, torch.tensor([[1., 2., 3., 4.]]))
        with self.assertRaises(ValueError):
            PMF.pooled_features(text, audio, vision, None)

    def test_train_statistics_checkpoint_and_single_batch(self):
        model = PMF(self.args())
        train = torch.tensor([[1., 2., 9., 4.], [3., 4., 9., 8.]])
        model.fit_normalization(train)
        torch.testing.assert_close(model.feature_mean, torch.tensor([2., 3., 9., 6.]))
        torch.testing.assert_close(model.feature_std, torch.tensor([1., 1., 1., 2.]))
        data = (torch.randn(1, 3, 2), torch.randn(1, 3, 1), torch.randn(1, 3, 1))
        mask = torch.tensor([[1., 1., 0.]])
        model.eval()
        out = model(*data, padding_mask=mask)
        self.assertEqual(out['polarity'].shape, (1, 3))
        self.assertEqual(out['magnitude'].shape, (1,))
        self.assertEqual(out['M'].shape, (1, 1))
        self.assertTrue(torch.all((out['magnitude'] > 0) & (out['magnitude'] < 3)))
        altered = tuple(x.clone() for x in data)
        for x in altered:
            x[:, 2, :] = 1e6
        other = model(*altered, padding_mask=mask)
        torch.testing.assert_close(out['polarity'], other['polarity'])
        torch.testing.assert_close(out['magnitude'], other['magnitude'])
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = PMF(self.args()).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        prediction = restored(*data, padding_mask=mask)
        torch.testing.assert_close(out['M'], prediction['M'])
        torch.testing.assert_close(model.feature_mean, restored.feature_mean)

    def test_two_head_gradients_and_parameter_count(self):
        model = PMF(self.args())
        model.fit_normalization(torch.randn(6, 4))
        out = model(torch.randn(3, 4, 2), torch.randn(3, 4, 1), torch.randn(3, 4, 1), padding_mask=torch.ones(3, 4))
        y = torch.tensor([-1., 0., 2.])
        mask = y != 0
        loss = torch.nn.functional.cross_entropy(out['polarity'], polarity_from_intensity(y))
        loss = loss + torch.nn.functional.l1_loss(out['magnitude'][mask], y[mask].abs())
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.zero_grad()
        out = model(torch.randn(3, 4, 2), torch.randn(3, 4, 1), torch.randn(3, 4, 1), padding_mask=torch.ones(3, 4))
        neutral_loss = torch.nn.functional.cross_entropy(out['polarity'], torch.ones(3, dtype=torch.long)) + out['magnitude'].sum()*0.0
        neutral_loss.backward()
        self.assertEqual(model.polmag_head.magnitude.weight.grad.abs().sum().item(), 0.)
        full = PMF(get_config_regression('pmf', 'mosei'))
        self.assertEqual(sum(p.numel() for p in full.parameters()), 162436)
        with self.assertRaises(ValueError):
            ATIO().getTrain(EasyDict(model_name='tfn', polmag=True))

    def test_result_schema_missing_corr_and_summary(self):
        from MMSA.utils.polmag_results import prepare_result_files,append_seed_results,append_summary
        with tempfile.TemporaryDirectory(prefix='pmf-csv-tests-') as tmp:
            tmp=Path(tmp)
            legacy=tmp/'mosei.csv'
            legacy.write_text('Model,old_metric\ntfn,66.25\n')
            before=legacy.read_bytes()
            detail,summary=prepare_result_files(tmp,'mosei')
            first={'Acc3':.7,'MacroF1':.6,'MAE':.52,'Corr':float('nan')}
            second={'Acc3':.8,'MacroF1':.8,'MAE':.72,'Corr':.4}
            results=[{'valid':m,'test':m,'test2':m,'best_epoch':2} for m in [first,second]]
            for seed,result in zip([1111,1112],results):
                append_seed_results(detail,'run-a','pmf',seed,'clean',result)
            append_summary(summary,'run-a','pmf',[1111,1112],'clean',results)
            with detail.open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),6)
            self.assertEqual(rows[0]['mae'],'0.52')
            self.assertEqual(rows[0]['corr'],'')
            with summary.open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),3)
            self.assertAlmostEqual(float(rows[0]['macro_f1_mean']),.7)
            self.assertAlmostEqual(float(rows[0]['macro_f1_std']),.1)
            self.assertEqual(rows[0]['corr_n'],'1')
            self.assertEqual(legacy.read_bytes(),before)
            detail.write_text('wrong,header\n')
            with self.assertRaisesRegex(ValueError,'Incompatible result CSV header'):
                prepare_result_files(tmp,'mosei')

    def test_framework_training_results_and_train_only_stats(self):
        run = importlib.import_module('MMSA.run')
        rng = np.random.default_rng(42)
        def split(n, shift):
            bert = np.zeros((n, 3, 50), dtype=np.int64)
            bert[:, 1, :4] = 1
            return {'text':rng.normal(shift, 1, (n, 50, 768)).astype('float32'),
                    'audio':rng.normal(shift, 1, (n, 50, 74)).astype('float32'),
                    'vision':rng.normal(shift, 1, (n, 50, 35)).astype('float32'),
                    'text_bert':bert, 'raw_text':np.asarray(['example']*n),
                    'id':np.asarray([str(i) for i in range(n)]),
                    'regression_labels':np.resize(np.asarray([-1.,0.,1.],dtype='float32'),n)}
        samples={'train':split(12,0), 'valid':split(6,20), 'test':split(6,-20)}
        test2={'test2':split(30,40)}
        expected=np.concatenate([samples['train'][key][:,:4].mean(axis=1) for key in ['text','audio','vision']],axis=-1).mean(axis=0)
        logger=logging.getLogger('MMSA');old_handlers=list(logger.handlers)
        try:
            with experiment_directory() as tmp:
                tmp=Path(tmp)
                for name,data in [('data.pkl',samples),('test2.pkl',test2)]:
                    with (tmp/name).open('wb') as stream:pickle.dump(data,stream)
                config={'device':'cpu', 'test2_feature':str(tmp/'test2.pkl'), 'early_stop':1, 'batch_size':12}
                with patch.object(run,'assign_gpu',side_effect=AssertionError('Explicit CPU must not probe GPUs')),patch.object(run.torch.cuda,'set_device',side_effect=AssertionError('CPU must not select a CUDA device')),patch.object(run.time,'sleep'):
                    run.MMSA_run('pmf','mosei',config=config,custom_feature=str(tmp/'data.pkl'),seeds=[1111,1112],num_workers=0,
                                 model_save_dir=tmp/'models',res_save_dir=tmp/'results',log_dir=tmp/'logs',verbose_level=0)
                weights=sorted((tmp/'models').glob('*/pmf-mosei-seed*.pth'))
                self.assertEqual(len(weights),2)
                self.assertNotEqual(weights[0],weights[1])
                self.assertTrue(all(w.with_suffix('.json').is_file() for w in weights))
                checkpoint=torch.load(weights[0],map_location='cpu',weights_only=True)
                np.testing.assert_allclose(checkpoint['Model.feature_mean'].numpy(),expected,atol=1e-6)
                self.assertTrue(checkpoint['Model.normalization_fitted'].item())
                with (tmp/'results/normal/mosei_polmag_v1.csv').open() as stream:rows=list(csv.DictReader(stream))
                self.assertEqual([row['split'] for row in rows],['valid','test','test2']*2)
                self.assertEqual({row['seed'] for row in rows},{'1111','1112'})
                self.assertTrue(all(row['model']=='pmf' for row in rows))
                self.assertEqual(rows[-1]['scenario'],'attachment3_missing_public_labels')
                self.assertTrue((tmp/'results/normal/mosei_polmag_v1_summary.csv').exists())
                text=(tmp/'logs/pmf-mosei.log').read_text()
                self.assertIn('VAL-(pmf)',text);self.assertIn('TEST2-(pmf)',text)
                feature={key:value[0] for key,value in samples['test'].items()}
                with (tmp/'single.pkl').open('wb') as stream:pickle.dump(feature,stream)
                args=get_config_regression('pmf','mosei')
                prediction=run.MMSA_test(str(weights[0].with_suffix('.json')),str(weights[0]),str(tmp/'single.pkl'),gpu_id=-1)
                self.assertTrue(np.isfinite(prediction))
                # Both portable CLIs run with only the supplied files.
                scripts = Path(__file__).resolve().parents[1] / 'scripts'
                stdout = io.StringIO()
                with patch('sys.argv', ['train_pmf.py','--data',str(tmp/'data.pkl'),'--test2',str(tmp/'test2.pkl'),'--device','cpu','--dry-run']), patch('sys.stdout', stdout):
                    runpy.run_path(str(scripts/'train_pmf.py'), run_name='__main__')
                report = json.loads(stdout.getvalue())
                self.assertEqual(report['splits'], {'train':12,'valid':6,'test':6,'test2':30})
                self.assertEqual(report['parameters'],162436)
                stdout = io.StringIO()
                with patch('sys.argv', ['predict_pmf.py','--config',str(weights[0].with_suffix('.json')),'--weights',str(weights[0]),'--features',str(tmp/'single.pkl')]), patch('sys.stdout', stdout):
                    runpy.run_path(str(scripts/'predict_pmf.py'), run_name='__main__')
                rows = json.loads(stdout.getvalue())
                self.assertEqual(len(rows),1)
                self.assertAlmostEqual(rows[0]['intensity'],prediction)
        finally:
            for handler in list(logger.handlers):
                if handler not in old_handlers:
                    logger.removeHandler(handler);handler.close()


if __name__ == '__main__':
    unittest.main()
