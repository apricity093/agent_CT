import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import torch
import numpy as np
import pytest
from inv_framework.solvers import ConstrainedTVSolver, constrained_pdhg, asd_pocs_tv, SolveControl
from inv_framework.ct_runtime import run_case, build_solver, validate_parameters
from inv_framework.instrumentation import CountingLinearOperator

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p4_example',ROOT/'examples/constrained_tv.py')
example=importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('limited',[False,True])
@pytest.mark.parametrize('name',['constrained_pdhg','asd_pocs_tv'])
def test_public_runtime(tmp_path,device,limited,name):
    case, op=example.make_case(tmp_path,limited)
    result=run_case(name,case.case_id,ROOT/f'configs/algorithms/{name}.yaml',tmp_path/'run',data_root=tmp_path,device=device,max_iterations=20,max_forward_calls=1500,max_adjoint_calls=1500)
    assert result['status']=='success',result
    assert result['convergence_status'] in ('max_iterations','stalled','converged')
    bundle=torch.load(tmp_path/'run/reconstruction.pt',weights_only=False)
    x=bundle['reconstruction']
    assert x.shape==(1,1,8,8) and x.dtype==torch.float32
    assert torch.isfinite(x).all() and (x>=0).all() and x.std()>0
    a=x.numpy()[0,0];dy=np.zeros_like(a);dx=np.zeros_like(a)
    dy[:-1]=a[1:]-a[:-1];dx[:,:-1]=a[:,1:]-a[:,:-1]
    tv=float(np.sqrt(dx*dx+dy*dy).sum())
    residual=float((op.forward(x)-case.measurement)[case.valid_measurement_mask].norm())
    d=result['diagnostics'];endpoint=d['convergence']['endpoint_confirmation']
    assert endpoint['passed']
    assert abs(endpoint['tv']-tv)<1e-5
    assert abs(endpoint['constraint_diagnostics']['raw_residual_norm']-residual)<1e-5
    assert endpoint['feasibility_status']==('feasible' if residual<=.20001 else 'violated')
    assert d['reconstruction_device'].startswith(device)
    assert d['parameters']['epsilon']==.2 and d['parameters']['num_iterations']==20
    assert d['resources']['forward_calls']<=1500 and d['resources']['adjoint_calls']<=1500

@pytest.mark.parametrize('name',['constrained_pdhg','asd_pocs_tv'])
def test_rejection(tmp_path,name):
    case,_=example.make_case(tmp_path)
    base={'epsilon':.2,'epsilon_source':'explicit'}
    for extra in ({'epsilon':-1},{'epsilon':float('nan')},{'epsilon_source':' '},{'strength':.1},{'regularizer':'h1'},{'data_fidelity':'wls'},{'min_value':None},{'max_value':1.},{'tv_mode':'bad'}):
        assert validate_parameters(name,{**base,**extra},case=case).errors
    from inv_framework.solvers.specs import validate_parameter_values
    for context in ({'geometry_type':'cone_3d'},{'observation_domain':'counts'},{'dimension':3}):
        assert validate_parameter_values(name,base,**context).errors

@pytest.mark.parametrize('name',['constrained_pdhg','asd_pocs_tv'])
def test_budget_and_cancel(tmp_path,name):
    case,op=example.make_case(tmp_path)
    result=run_case(name,case.case_id,ROOT/f'configs/algorithms/{name}.yaml',tmp_path/'budget',data_root=tmp_path,max_forward_calls=2)
    assert result['status']=='resource_exhausted'
    p=validate_parameters(name,{'epsilon':.2,'epsilon_source':'explicit','num_iterations':3},case=case).parameters
    solver=build_solver(name,p,case)
    r=solver.solve_detailed(case.measurement,op,mask=case.valid_measurement_mask,callback=lambda record:False)
    assert r.status=='cancelled' and r.actual_iterations==1

@pytest.mark.parametrize('name',['constrained_pdhg','asd_pocs_tv'])
def test_numerical_failure(tmp_path,monkeypatch,name):
    case,op=example.make_case(tmp_path)
    import inv_framework.ct_runtime as runtime
    def bad(x):return torch.full_like(case.measurement,float('nan'))
    monkeypatch.setattr(op,'forward',bad)
    monkeypatch.setattr(runtime,'build_operator',lambda *a:op)
    result=run_case(name,case.case_id,ROOT/f'configs/algorithms/{name}.yaml',tmp_path/'bad',data_root=tmp_path)
    assert result['status']=='numerical_error',result


def test_cli(tmp_path):
    case,_=example.make_case(tmp_path)
    command=[sys.executable,'-m','inv_framework.cli','run','constrained_pdhg','--case',case.case_id,'--data-root',str(tmp_path),'--config',str(ROOT/'configs/algorithms/constrained_pdhg.yaml'),'--out',str(tmp_path/'cli'),'--max-forward-calls','1500','--max-adjoint-calls','1500']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((tmp_path/'cli/diagnostics.json').read_text())['convergence_status']=='max_iterations'


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('mode',['isotropic','anisotropic'])
def test_python_exports(tmp_path,device,mode):
    case,base=example.make_case(tmp_path)
    from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
    op=ParallelBeamRadon2D(8,angles=torch.tensor(case.geometry['angles_rad'],device=device),device=device)
    y=case.measurement.to(device=device,dtype=torch.float64)
    mask=case.valid_measurement_mask.to(device)
    for fn in (constrained_pdhg,asd_pocs_tv):
        x=fn(y,op,mask=mask,epsilon=.2,epsilon_source='explicit export fixture',tv_mode=mode,num_iterations=2)
        assert x.dtype==y.dtype and x.device==y.device
        assert x.shape==(1,1,8,8) and torch.isfinite(x).all() and (x>=0).all()


@pytest.mark.parametrize('name',['constrained_pdhg','asd_pocs_tv'])
def test_native_public_status_and_options(tmp_path,name):
    case,op=example.make_case(tmp_path)
    options={'epsilon':.31,'epsilon_source':'explicit public options','num_iterations':40,
             'tv_mode':'anisotropic','feasibility_tolerance':2e-5}
    if name=='asd_pocs_tv':
        options.update(relaxation=.8,relaxation_decay=.97,tv_step=.1,tv_step_decay=.9,
                       tv_updates=2,update_ratio=.8,block_size=2)
    validation=validate_parameters(name,options,case=case)
    assert not validation.errors
    solver=build_solver(name,validation.parameters,case)
    for key,value in options.items():assert solver.parameters[key]==value
    r=solver.solve_detailed(torch.zeros_like(case.measurement),op,mask=case.valid_measurement_mask,
        control=SolveControl(min_iterations=10,check_every=2,patience=7))
    assert r.actual_iterations==22
    assert r.status==('converged' if name=='constrained_pdhg' else 'stalled')

@pytest.mark.parametrize('name',['constrained_pdhg','asd_pocs_tv'])
def test_runtime_rejects_before_operator(tmp_path,monkeypatch,name):
    case,_=example.make_case(tmp_path)
    import inv_framework.ct_runtime as runtime
    def forbidden(*args):raise AssertionError('operator constructed before invalid combination rejection')
    monkeypatch.setattr(runtime,'build_operator',forbidden)
    config=tmp_path/'invalid.yaml'
    config.write_text(f'name: {name}\nparameters:\n  epsilon: 0.2\n  epsilon_source: explicit\n  strength: 0.1\n')
    result=run_case(name,case.case_id,config,tmp_path/'invalid',data_root=tmp_path)
    assert result['status']=='invalid_parameters'


@pytest.mark.parametrize('device',DEVICES)
def test_float64_ct_adjoint(device):
    from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D, _radon_forward_impl
    angles=torch.tensor([0.,.13,.47,.81,1.52,2.17,2.91],device=device)
    op=ParallelBeamRadon2D(8,angles=angles,device=device)
    torch.manual_seed(17)
    x=torch.randn(1,1,8,8,device=device,dtype=torch.float64,requires_grad=True)
    y=torch.randn(1,1,7,8,device=device,dtype=torch.float64)
    ax=_radon_forward_impl(x,angles)
    adj=op.adjoint(y)
    native_grad,=torch.autograd.grad((ax*y).sum(),x)
    assert abs(float((ax*y).sum()-(x*adj).sum()))<1e-10
    torch.testing.assert_close(adj,native_grad,atol=1e-10,rtol=1e-10)
