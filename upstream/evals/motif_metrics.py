"""Report the repository and paper motif success definitions independently."""
import numpy as np
import pandas as pd


def motif_success_statistics(eval_df, problem_names, tm_thr=0.8, motif_rmsd_thr=1.0):
    annotated = eval_df.copy()
    problem_names = list(problem_names)
    if not set(annotated['pdb_name']).issubset(problem_names):
        raise ValueError('Evaluation contains unknown motif problems')
    protocols = {
        'repository': {
            'columns': ['tm_score', 'pred_motif_rmsd'],
            'success': (annotated['tm_score'] > tm_thr) & (annotated['pred_motif_rmsd'] < motif_rmsd_thr),
            'criterion': f'scTM > {tm_thr} and refolded motif RMSD < {motif_rmsd_thr} A',
        },
        'paper': {
            'columns': ['rmsd', 'gen_motif_rmsd'],
            'success': (annotated['rmsd'] < 2.0) & (annotated['gen_motif_rmsd'] < 1.0),
            'criterion': 'backbone scRMSD < 2.0 A and generated motif RMSD < 1.0 A',
        },
    }
    summaries, problems = [], []
    for protocol, definition in protocols.items():
        valid = np.isfinite(annotated[definition['columns']].to_numpy(dtype=float)).all(axis=1)
        annotated[f'success_{protocol}'] = definition['success'] & valid
        passed = annotated[annotated[f'success_{protocol}']]
        summaries.append({
            'protocol': protocol, 'criterion': definition['criterion'],
            'n_samples': len(annotated), 'n_valid': int(valid.sum()),
            'n_success': len(passed), 'n_solved': passed['pdb_name'].nunique(),
            'n_problems': len(problem_names),
            'success_rate': len(passed) / len(annotated) if len(annotated) else np.nan,
        })
        for name in problem_names:
            group = annotated[annotated['pdb_name'] == name]
            n_success = int(group[f'success_{protocol}'].sum())
            problems.append({
                'protocol': protocol, 'pdb_name': name, 'n_samples': len(group),
                'n_success': n_success, 'solved': n_success > 0,
                'success_rate': n_success / len(group) if len(group) else np.nan,
            })
    return annotated, pd.DataFrame(summaries), pd.DataFrame(problems)
