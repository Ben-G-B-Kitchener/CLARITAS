from __future__ import annotations
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CONFIG_NAME='claritas_v24_36_12_config.json'
_cfg=json.loads((HERE/CONFIG_NAME).read_text())
RELEASE=str(_cfg['version'])
PREFIX='v'+RELEASE[1:].replace('.','_')
PACKAGE='CLARITAS_'+RELEASE.replace('.','_')
DEFAULT_RESULTS='results'
AUDIT_SUBDIR='_reciprocity_fallback_audit'
RECEIPT_NAME='.v24_36_12_gpu_receipt.json'
OUTPUT_FILENAMES=(
    f'{PREFIX}_fixed_microfacet_reversibility.csv',
    f'{PREFIX}_complete_kernel_reciprocity_pairs.csv',
    f'{PREFIX}_complete_kernel_reciprocity_summary.csv',
    f'{PREFIX}_fallback_flux_decomposition.csv',
    f'{PREFIX}_macro_tir_reciprocal_flux.csv',
    f'{PREFIX}_probability_conservation.csv',
    f'{PREFIX}_process_decision.csv',
    f'{PREFIX}_process_decision.json',
    f'{PREFIX}_acquisition_summary.json',
    f'{PREFIX}_complete_kernel_forward_reverse.png',
    f'{PREFIX}_complete_kernel_residuals.png',
    f'{PREFIX}_fallback_flux_decomposition.png',
)
