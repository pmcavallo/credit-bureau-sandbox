import argparse, yaml, json, datetime as dt

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config/base.yaml'); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config,'r'))
    plan={
        'name':'cbsandbox_pipeline',
        'generated_utc':dt.datetime.utcnow().isoformat()+'Z',
        'artifacts_from_manifest':'outputs/reports/handoff_manifest.json',
        'steps':[
            {'name':'validate_contract','type':'processing','entry':'contract_validation','inputs':['data/bureau_sample.csv'],'outputs':['outputs/reports/contract_validation.json']},
            {'name':'train','type':'training','entry':'scripts/train_local.py','outputs':['outputs/metrics/metrics.json','outputs/models/MODEL_VERSION.txt']},
            {'name':'evaluate','type':'evaluation','entry':'scripts/evaluate.py','outputs':['outputs/metrics/evaluation.json']},
            {'name':'explain','type':'processing','entry':'shap_reason_codes','outputs':['outputs/reports/shap_global_importance.csv','outputs/reports/top_reasons_sample.csv','outputs/reports/reason_code_candidates.json']},
            {'name':'auditlog','type':'processing','entry':'audit_log_sample','outputs':['outputs/reports/audit_log_sample.jsonl']},
            {'name':'register','type':'conditional','condition':'auc_valid >= 0.65 and ks_valid >= 0.20 and psi_valid_vs_train < 0.10','registry_target':'local|sagemaker'}
        ],
        'notes':'This is a dry-run spec for SageMaker. We will map each step to Processing/Training/Evaluation/Model Registry when AWS creds + S3 bucket are configured.'
    }
    print(json.dumps(plan,indent=2))

if __name__=="__main__":
    main()
