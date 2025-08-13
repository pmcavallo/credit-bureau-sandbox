import argparse, json, os, re, boto3
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionLessThan

def parse_s3_uri(uri: str):
    m = re.match(r'^s3://([^/]+)/(.+)$', uri)
    if not m:
        raise ValueError(f'Bad S3 URI: {uri}')
    return m.group(1), m.group(2)

def build_pipeline(pipeline_session, pipeline_name: str):
    AUC = ParameterFloat(name='AUC')
    KS  = ParameterFloat(name='KS')
    PSI = ParameterFloat(name='PSI')
    AUC_T = ParameterFloat(name='AUC_T', default_value=0.65)
    KS_T  = ParameterFloat(name='KS_T',  default_value=0.20)
    PSI_T = ParameterFloat(name='PSI_T', default_value=0.10)
    fail = FailStep(name='MetricsGateFailed', error_message='AUC/KS/PSI did not meet thresholds.')
    cond = ConditionStep(
        name='MetricsGate',
        conditions=[
            ConditionGreaterThanOrEqualTo(left=AUC, right=AUC_T),
            ConditionGreaterThanOrEqualTo(left=KS, right=KS_T),
            ConditionLessThan(left=PSI, right=PSI_T),
        ],
        if_steps=[],
        else_steps=[fail],
    )
    return Pipeline(
        name=pipeline_name,
        parameters=[AUC, KS, PSI, AUC_T, KS_T, PSI_T],
        steps=[cond],
        sagemaker_session=pipeline_session,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', required=True)
    ap.add_argument('--bucket', required=False)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--metrics_s3', required=True)
    args = ap.parse_args()
    session = boto3.Session()
    region = session.region_name or os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION') or 'us-east-1'
    pipeline_session = PipelineSession(boto_session=session)
    bkt, key = parse_s3_uri(args.metrics_s3)
    s3 = session.client('s3', region_name=region)
    obj = s3.get_object(Bucket=bkt, Key=key)
    m = json.loads(obj['Body'].read().decode('utf-8'))
    auc = float(m['auc']); ks = float(m['ks']); psi = float(m['psi'])
    pipe_name = f'{args.prefix}-metrics-gate'
    pipeline = build_pipeline(pipeline_session, pipe_name)
    pipeline.upsert(role_arn=args.role)
    print(f'Starting pipeline: {pipe_name} with AUC={auc}, KS={ks}, PSI={psi}')
    exe = pipeline.start(parameters={'AUC': auc, 'KS': ks, 'PSI': psi})
    print('PipelineExecutionArn:', exe.arn)
    exe.wait(max_attempts=120)
    desc = exe.describe()
    print('Status:', desc.get('PipelineExecutionStatus'))
    if desc.get('FailureReason'):
        print('FailureReason:', desc['FailureReason'])

if __name__ == '__main__':
    main()
