"""Integrity regression at the real generated-plan/workspace-filter seam."""
import copy
import json
from pathlib import Path
import unittest
from dataclasses import replace

from experiments.exp_100a.run_experiment import search_integrity_failures as check_integrity, search_shape_block, validate_config
from vortex_runtime.explicit_rectangular_fmm import orientations
from test_explicit_rectangular_fmm import strassen_222

def search_integrity_failures(searches, **kwargs):
    return check_integrity(searches, search_contract={
        'minimum_depth':1,'maximum_depth':2,'static_scalar_bytes':[2,4]}, **kwargs)


class WorkspaceIntegrityTests(unittest.TestCase):
    def fixture(self, workspace=1):
        config=json.loads(Path('experiments/exp_100a/config.json').read_text(encoding='utf-8'))
        # Unit fixture only: deliberately tiny memory screen drives the same
        # empty-frontier seam as the frozen 16384x16384x128256 experiment.
        config=copy.deepcopy(config)
        config['target_contract']['peak_vram_bytes']=workspace
        config['search'].update(maximum_depth=2,beam_width=8,maximum_structural_plans_per_shape=8)
        schemes=orientations(strassen_222())
        result=search_shape_block(block_length=8,rows=8,columns=8,schemes=schemes,
            scheme_map={row.scheme_id:row for row in schemes},config=config)
        return result

    def test_generated_plans_all_filtered_by_budget_are_not_corruption(self):
        result=self.fixture()
        self.assertFalse(result.direct_structural_plans)
        self.assertTrue(result.oracle_structural_plans)
        self.assertEqual(search_integrity_failures({'unit':result}),[])

    def test_empty_collection_and_missing_registered_shape_are_invalid(self):
        self.assertIn('empty_shape_search_collection',search_integrity_failures({}))
        self.assertIn('missing_or_unregistered_shape_search',search_integrity_failures(
            {'unit':self.fixture()},expected_keys={'unit','missing'}))

    def test_lost_feasible_frontier_is_invalid(self):
        result=self.fixture(1<<30)
        self.assertTrue(result.direct_structural_plans)
        broken=replace(result,direct_structural_plans=())
        self.assertIn('incomplete_or_inconsistent_shape_search_accounting',
            search_integrity_failures({'unit':broken}))

    def test_incomplete_missing_and_inconsistent_accounting_are_invalid(self):
        result=self.fixture()
        mutations=[None,dict(result.accounting,retained_search_completed=False),
            dict(result.accounting,workspace_rejected_count=0),
            dict(result.accounting,minimum_generated_workspace_bytes=0)]
        for audit in mutations:
            with self.subTest(audit=audit):
                self.assertIn('incomplete_or_inconsistent_shape_search_accounting',
                    search_integrity_failures({'unit':replace(result,accounting=audit)}))

    def test_zero_caps_are_invalid(self):
        config=json.loads(Path('experiments/exp_100a/config.json').read_text(encoding='utf-8'))
        config['search']['maximum_structural_plans_per_shape']=0
        with self.assertRaisesRegex(RuntimeError,'search cap must be positive'):
            validate_config(config)

    def test_paired_undercount_partial_depth_and_oracle_are_invalid(self):
        result=self.fixture()
        a=dict(result.accounting)
        changes=[
            dict(evaluated_direct_plan_count=a['evaluated_direct_plan_count']-1,
                 workspace_rejected_count=a['workspace_rejected_count']-1),
            dict(completed_evaluated_depths=[1]),
            dict(termination='unexplained'),
            dict(evaluated_oracle_plan_count=0),
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assertIn('incomplete_or_inconsistent_shape_search_accounting',
                    search_integrity_failures({'unit':replace(result,accounting=dict(a,**change))}))


if __name__=='__main__':
    unittest.main()
