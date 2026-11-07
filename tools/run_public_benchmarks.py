"""Run the primary lightweight standard benchmarks; no tuning or automatic retries."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from benchmark_runs import ROOT, Bench, task, read_cases
import matplotlib.pyplot as plt


def run(output,checkpoint):
    torch.set_num_threads(1)
    bench=Bench(output)
    cases={d:[c for c in read_cases(d) if c.get('benchmark_group')=='public_standard'] for d in ('mri','deconvolution')}
    for case in cases['mri']:
        radial=case['model']['sampling']=='noncartesian'
        for algorithm in (['cg_sense'] if radial else ['zero_filled','cg_sense','cs_tv','cs_wavelet','grappa']):
            params=dict(lam=.001,max_iter=100) if algorithm=='cg_sense' else dict(lam=.005,max_iter=60,cg_max_iter=30) if algorithm.startswith('cs') else dict(kernel_width=7) if algorithm=='grappa' else {}
            bench.run('mri',case,algorithm,params,label='standard')
        if radial:continue
        bench.run('mri',case,'espirit',dict(crop=.8,svd_threshold=.02),label='standard')
        config=task('mri',case,'cg_sense',dict(lam=.001,max_iter=100))
        config['inputs']['sensitivity_maps']=dict(path=str(output/'standard/mri'/case['case_id']/'gaussian_espirit/sensitivity_maps.npy'))
        bench.run('mri',case,'cg_sense',label='calibrated',config=config)
        if case['split']=='eval':bench.run('mri',case,'varnet',dict(checkpoint=str(checkpoint),num_low_frequencies=case['num_low_frequencies']),label='phantom_inference_only')
    for case in cases['deconvolution']:
        for noise,algorithms in [('gaussian',['wiener_hunt','tv_admm','hyper_laplacian']),('poisson',['richardson_lucy','tv_admm','rl_tv'])]:
            for algorithm in algorithms:
                params=dict(balance=.01) if algorithm=='wiener_hunt' else dict(max_iter=20) if algorithm=='richardson_lucy' else dict(lam=.05,max_iter=50) if algorithm=='rl_tv' else dict(lam=.03 if noise=='poisson' else .005,max_iter=200) if algorithm=='tv_admm' else dict(lam=.001,p=.5,max_iter=60)
                bench.run('deconvolution',case,algorithm,params,noise,label='standard')
        if case['object_kind']=='cell':continue
        for algorithm in ('blind_dcp','blind_l0'):
            params=dict(max_iter=3,final_max_iter=60)
            if algorithm=='blind_dcp':params.update(patch_size=7,latent_iter=6,cg_max_iter=50)
            bench.run('deconvolution',case,algorithm,params,label='blind')
    rows=json.loads((output/'summary.json').read_text())
    write_configs(output,rows)
    write_data_atlas(output)
    report(output)


def write_data_atlas(output):
    # One compact atlas makes the observation/reference/PSF distinction visible.
    selected=[c for c in read_cases('deconvolution') if c.get('benchmark_group')=='public_standard']
    fig,axes=plt.subplots(len(selected),4,figsize=(10,7))
    for panels,case in zip(axes,selected):
        folder=ROOT/'tests/data/deconvolution'/case['path'];public=np.load(folder/'public.npz');reference=np.load(folder/'reference.npz')['object']
        for ax,image,title in zip(panels,[reference,public['gaussian'],(public['counts']-.2)/40,public['psf']],
                                 [case['object_kind']+' reference','Gaussian observation','Poisson minus b / gain','true PSF (nonblind only)']):
            ax.imshow(image,cmap='gray',vmin=0,vmax=None if 'PSF' in title else 1);ax.set_title(title,fontsize=8);ax.axis('off')
    fig.tight_layout();fig.savefig(output/'data_atlas.png',dpi=150);plt.close(fig)


def write_configs(output,rows):
    """Export editable examples after a normal benchmark run."""
    for r in rows:
        if r['run'].startswith(('standard/','blind/','phantom_inference_only/')):
            source=output/r['run']/'task.yaml'
            config=yaml.safe_load(source.read_text())
            destination=ROOT/'configs'/r['domain']/'benchmarks'/f'standard_{r["case"]}_{r["noise"]}_{r["algorithm"]}.yaml'
            def relative(value):
                if isinstance(value,dict):return {k:os.path.relpath((source.parent/v).resolve(),destination.parent).replace('\\','/') if k in ('path','checkpoint') else relative(v) for k,v in value.items()}
                return value
            destination.write_text(yaml.safe_dump(relative(config),sort_keys=False),encoding='utf8')


def report(output):
    """Read saved measurements and write only report.md; never execute a task."""
    rows=json.loads((output/'summary.json').read_text())
    lines=['# 标准轻量数据基准结果','','当前 MRI 与反卷积使用 public_standard 数据组，算法与数据对应关系见仓库 docs/public_benchmarks.md。', '',
           '![公开图像 生成观测 真实PSF](data_atlas.png)','',
           '自然图像来自 scikit-image；退化观测仍为合成周期卷积。Levin 核取自 DPIR 的 Levin09.mat；Airy 为理想二维衍射模型。MR 数据为 phantominator 标准组织参数模体，加显式 Biot–Savart 环形线圈和独立 FFT/DFT 观测，仍是模拟采集。', '',
           '参数为固定示例，无针对评价集搜索或参数敏感性实验。原始尺度 data_range=1；MRI 按幅度评价，Poisson 输出按已知 gain40 转回对象尺度；没有真值强度拟合或核对齐。原始图像、PSF、模型和来源随案例保存。', '',
           '质量指标来自 summary.json 和各次 evaluation.json，实际有效参数与执行状态来自 result.json。下表秒数为 API runtime_sec，不含独立评价、制图和文件保存；wall_sec、设备和精度另随每次记录列出。各残差属于相应观测域，不跨噪声模型直接比较。', '',
           'VarNet 的模体调用仅为推理流程示例，单列 phantom_inference_only；它不是膝关节临床质量测试。真实 fastMRI 数据需要用户取得授权的本地完整采集，禁止将官方随机 mock fixture 标成真实数据，也不承诺35 dB。', '',
           '| 实验组 | 案例 | 方法 | PSNR dB | SSIM | RMSE | NMSE | 数据残差 | 核相对误差 | 秒 | status | converged | 步数和停止 |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
    for r in rows:
        show=lambda key:'—' if r.get(key) is None else f'{r[key]:.6g}'
        lines.append(f'| {r["run"].split("/")[0]}/{r["noise"]} | {r["case"]} | {r["algorithm"]} | {show("psnr")} | {show("ssim")} | {show("rmse")} | {show("nmse")} | {show("data_residual")} | {show("kernel_relative_error")} | {show("runtime_sec")} | {r["status"]} | {str(r["converged"]).lower()} | {r["iterations"]}/{r["stop"]} |')
    lines+=['','## 执行状态与适用范围','',f'共 {len(rows)} 次调用，{sum(r["status"]=="success" for r in rows)} 次成功执行，{sum(r["converged"] for r in rows)} 次标记为收敛，{sum(r["stop"]=="max_iterations" for r in rows)} 次达到迭代上限。上限不等于收敛；completed_valid 表示直接或固定网络执行完成，不标为优化收敛。已知真实核的非盲结果也不是理论上界。','']
    failures=[r for r in rows if r['status']!='success']
    if not failures:lines+=['本批记录没有失败调用；未收敛任务及其停止原因完整保留在表中。','']
    for r in failures:
        saved=json.loads((output/r['run']/'result.json').read_text())
        lines+=[f'- {r["run"]}：{json.dumps(saved.get("error"),ensure_ascii=False)}','']
    lines+=['这是有限标准样例上的模型匹配实验：自然图像退化不等于实拍模糊图像对，cell 是相位图的强度模板，理想 Airy 不包含完整显微镜效应。MR 模体不验证临床泛化；有限迭代和非凸盲估计没有全局最优保证。较高 PSNR 不证明理论正确。参数由 YAML/API 显式配置，Agent 可自行决定后续调用，本报告没有新增参数搜索。','','## 图像','']
    for r in rows:
        if 'psnr' in r and r['split']=='eval':
            lines += [f'### {r["run"]}','',f'![真值 重建 误差]({r["run"]}/comparison.png)','']
            if 'kernel_rmse' in r:lines += [f'![真实与估计 PSF]({r["run"]}/kernel.png)','']
    lines+=['## 各次实际参数与原始记录','']
    for r in rows:
        saved=json.loads((output/r['run']/'result.json').read_text())
        lines += [f'### {r["run"]}','',
                  f'[任务配置]({r["run"]}/task.yaml) · [执行记录]({r["run"]}/result.json) · [独立评价]({r["run"]}/evaluation.json)','',
                  f'设备 {r["device"]}，精度 {r["dtype"]}；API runtime_sec={r["runtime_sec"]}，runner wall_sec={r["wall_sec"]}。','',
                  '实际 effective_params：','','```json',json.dumps(saved['effective_params'],indent=2,ensure_ascii=False),'```','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/public_benchmarks')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/weights/knee_leaderboard_state_dict.pt')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:report(args.output.resolve())
    else:run(args.output.resolve(),args.checkpoint.resolve())
