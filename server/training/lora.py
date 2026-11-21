"""Opt-in registered LoRA recipe. Client configuration contains no filesystem paths."""
import math
import re
from version.v8.training.lora_package import load_package

RECIPE = 'decoder_lora_fp32_v1'


def packages(registry):
    from .contract import ident
    if type(registry) is not dict or not 1 <= len(registry) <= 8:
        raise ValueError('invalid LoRA registry')
    result = {}
    for name, entry in registry.items():
        if not ident(name) or set(entry) != {'path', 'sha256'}:
            raise ValueError('invalid LoRA registration')
        doc, _, discovery = load_package(entry['path'], entry['sha256'])
        result[name] = dict(id=name, sha256=entry['sha256'], tokens=doc['tokens'],
                            base={k:v for k,v in doc['base'].items() if k != 'directory'},
                            tokenizer=doc['tokenizer'], targets=discovery['targets'],
                            splits={s:sum(r['split']==s for r in doc['records']) for s in ('train','validation')})
    return result


def validate(config, available):
    from .contract import ident
    errors = []
    def error(field, code): errors.append(dict(field='/config/'+field, code=code))
    fields = {'recipe','package_id','package_sha256','dtype','optimizer','method','steps','learning_rate','adapters'}
    for field in sorted(fields-set(config)): error(field,'REQUIRED')
    for field in sorted(set(config)-fields): error(field,'UNKNOWN_FIELD')
    for field, value in [('recipe',RECIPE),('dtype','fp32'),('optimizer','adamw'),('method','lora')]:
        if config.get(field) != value: error(field,'UNSUPPORTED_VALUE')
    package = available.get(config.get('package_id')) if type(config.get('package_id')) is str else None
    if package is None: error('package_id','UNKNOWN_PACKAGE')
    elif config.get('package_sha256') != package['sha256']: error('package_sha256','IDENTITY_MISMATCH')
    if type(config.get('steps')) is not int or not 1 <= config['steps'] <= 100000: error('steps','OUT_OF_RANGE')
    lr = config.get('learning_rate')
    if type(lr) not in (int,float) or not math.isfinite(lr) or not 0 < lr <= .01: error('learning_rate','OUT_OF_RANGE')
    rows = config.get('adapters')
    if type(rows) is not list or not 1 <= len(rows) <= 16:
        error('adapters','OUT_OF_RANGE'); return errors
    names, targets, count = set(), set(), 0
    available_targets = {r['target']:r for r in package['targets']} if package else {}
    for row in rows:
        if type(row) is not dict or set(row) != {'target','rank','alpha','name','seed'}:
            error('adapters','INVALID_ADAPTER'); continue
        if type(row['name']) is not str or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,47}',row['name']) or row['name'] in names: error('adapters','INVALID_NAME')
        else: names.add(row['name'])
        target = available_targets.get(row['target']) if type(row['target']) is str else None
        if target is None or row['target'] in targets: error('adapters','INVALID_TARGET')
        else: targets.add(row['target'])
        if type(row['rank']) is not int or target is None or not target['rank_min'] <= row['rank'] <= target['rank_max']: error('adapters','INVALID_RANK')
        else: count += row['rank']*target['parameters_per_rank']
        if type(row['alpha']) not in (int,float) or not math.isfinite(row['alpha']) or not 0 < row['alpha'] <= 1024: error('adapters','INVALID_SCALE')
        if type(row['seed']) is not int or not 0 <= row['seed'] < 2**32: error('adapters','INVALID_SEED')
    if count > 1000000: error('adapters','PARAMETER_BUDGET_EXCEEDED')
    return errors


def validate_result(report, config, package):
    """Fail closed on incomplete/misattributed job evidence; no model execution."""
    from .contract import digest
    import re
    if type(report) is not dict:
        return False
    if (report.get('schema') != 'cke.lora.training-result.v1'
            or report.get('package_sha256') != config['package_sha256']
            or report.get('config_sha256') != digest(config)
            or type(report.get('completed_steps')) is not int
            or report['completed_steps'] != config['steps']):
        return False
    expected = {r['id'] for r in package['records'] if r['split']=='validation'}
    metrics = report.get('metrics', {})
    validation = report.get('validation', {})
    if type(metrics) is not dict or type(validation) is not dict:
        return False
    for phase, metric in [('before','validation_loss_before'),('after','validation_loss_after')]:
        group = validation.get(phase)
        if type(group) is not dict or set(group) != {'loss','records'}:
            return False
        rows = group['records']
        if type(rows) is not list or len(rows) != len(expected):
            return False
        seen = set()
        for row in rows:
            if type(row) is not dict or set(row) != {'id','loss','logits_sha256'}:
                return False
            if type(row['id']) is not str or row['id'] not in expected or row['id'] in seen:
                return False
            seen.add(row['id'])
            if type(row['loss']) not in (int,float) or not math.isfinite(row['loss']) or row['loss'] < 0:
                return False
            if type(row['logits_sha256']) is not str or not re.fullmatch('[a-f0-9]{64}',row['logits_sha256']):
                return False
        loss = sum(r['loss'] for r in rows)/len(rows)
        if not math.isfinite(loss) or type(group['loss']) not in (int,float) or type(metrics.get(metric)) not in (int,float) or group['loss'] != loss or metrics.get(metric) != loss:
            return False
    losses = report.get('training_losses')
    if (type(losses) is not list or len(losses) != config['steps']
            or not all(type(x) in (int,float) and math.isfinite(x) and x >= 0 for x in losses)
            or type(metrics.get('training_loss_last')) not in (int,float)
            or metrics.get('training_loss_last') != losses[-1]):
        return False
    verdicts = report.get('verdicts')
    return verdicts == dict(execution='PASS', reload='PASS', numerical_oracle='NOT_TESTED',
                            held_out_quality='MEASURED_NOT_CERTIFIED', base_provenance=package['base']['provenance'])
