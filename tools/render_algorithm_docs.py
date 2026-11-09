"""Build one manual per numerical method, retaining domain entry contracts."""
from pathlib import Path
from collections import defaultdict
from inv_framework.algorithm_catalog import algorithm_documentation

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {'ct': 'CT', 'mri': 'MRI', 'deconvolution': '反卷积'}
PARAMETER_HELP = {
 'boundary':'正则差分边界；物理卷积边界由model单独声明',
 'penalize_lowpass':'是否惩罚Haar最终低频系数',
 'haar_strength':'联合目标中的Haar权重，独立于TV的strength',
 'rho_data':'ADMM数据分支罚参数；null继承rho',
 'rho_regularizer':'ADMM主正则分支罚参数；null继承rho',
 'rho_haar':'ADMM联合Haar分支罚参数；null继承rho',
 'rho_constraint':'ADMM恒等约束分支罚参数；null继承rho',
 'subproblem_solver':'线性子问题解法；CT使用CG或小系统Cholesky，周期卷积可用FFT',
}
# title, purpose, source module, scope/variant description
METHODS = {
 'admm': ('ADMM','通过数据、正则与约束分裂求解重建目标。','solvers/composition.py','CS-TV、CS-Wavelet、TV-ADMM 是领域专用入口；TV、Haar、TV＋Haar 是目标配置，不另计算法。'),
 'pdhg': ('PDHG','使用原始—对偶更新求解可分裂目标。','solvers/splitting.py','constrained_pdhg 是残差球约束目标的专用入口。自动步长依赖实际分裂算子上界。'),
 'fista': ('FISTA','使用加速近端梯度和光滑项回溯执行重建。','solvers/splitting.py','tv_fista 是 TV 专用入口；近端精度、边界和约束应与实际目标一致。'),
 'landweber': ('Landweber','使用增广算子及伴随进行最小二乘梯度迭代。','solvers/quadratic.py','自动步长依据加权增广算子；CT 有界旧入口采用投影梯度。'),
 'cgls': ('CGLS','以最小二乘共轭方向求解实数或复数二次目标。','solvers/quadratic.py','恒等与梯度二次正则通过增广实现；公共共享路径为无约束目标。'),
 'lsqr': ('LSQR','使用 Golub–Kahan 双对角化求解最小二乘。','solvers/quadratic.py','damping 的平方进入恒等正则，不能与显式正则重复计入。'),
 'cg': ('共轭梯度 CG','求解 Hermitian 正定正规系统。','core/krylov.py','cg_sense 是 SENSE 入口；CT tikhonov 是采用 CG 的二次正则入口。Tikhonov 为目标项，不另计算法。独立 cg 要求正权重恒等正则。'),
 'fbp': ('滤波反投影 FBP','对投影滤波后反投影得到图像。','solvers/analytic.py','用于二维平行束线积分；锥束体重建采用 FDK。'),
 'fdk': ('锥束滤波反投影 FDK','调用 ASTRA FDK_CUDA 重建圆轨迹锥束数据。','ct/backends','要求受支持的 cone 几何、等边体素、ASTRA CUDA 与 PyTorch CUDA。'),
 'sirt': ('SIRT','用行列归一化的同时残差更新执行代数重建。','solvers/algebraic.py','二维平行束 CT，可设置图像上下界。'),
 'sart': ('SART','按角度块执行归一化代数更新。','solvers/algebraic.py','OS-SART 作为有序子集变体合并说明；子集与块大小须符合实际角度划分。'),
 'mlem': ('最大似然 EM（MLEM）','以非负乘法更新求解发射 Poisson 计数重建。','solvers/statistical.py','OSEM 是有序子集变体。发射计数与 X-ray 透射计数是不同模型，不对透射数据套用该更新。'),
 'asd_pocs_tv': ('ASD-POCS-TV','交替执行观测可行性更新和 TV 下降。','solvers/splitting.py','这是独立的投影与下降更新方法；需要明确的残差球半径及来源，不是 ADMM 的 TV 配置。'),
 'zero_filled': ('Zero-filled','对采样 k-space 作逆 FFT 并组合线圈图像。','mri/api.py','SENSE 组合输出复数，RSS 输出幅值；参考图应匹配线圈组合约定。'),
 'grappa': ('GRAPPA','从 ACS 拟合插值核并补全缺失 k-space。','mri/calibration.py','二维 Cartesian 多线圈；需要实际 ACS、匹配的采样掩模和核宽。'),
 'espirit': ('ESPIRiT','从 ACS 子空间估计线圈灵敏度图。','mri/calibration.py','这是校准任务，输出灵敏度及特征值；重建应另发显式请求。'),
 'varnet': ('VarNet','使用训练得到的级联网络进行多线圈 MRI 重建。','mri/varnet_inference.py','二维 Cartesian 输入；网络结构和 ACS 设置须匹配本地权重。'),
 'wiener_hunt': ('Wiener–Hunt','频域求解已知核的 Laplacian 二次正则目标。','deconvolution/solvers.py','周期卷积、已知 PSF、二次数据项；balance 为 Laplacian 正则权重。'),
 'richardson_lucy': ('Richardson–Lucy','通过 Poisson 似然乘法更新恢复非负图像。','deconvolution/solvers.py','RL-TV 是带平滑 TV 的 Dey 更新变体，在本家族说明；需满足其分母条件。'),
 'hyper_laplacian': ('Hyper-Laplacian 反卷积','通过半二次分裂求解非凸梯度先验。','deconvolution/regularized.py','周期已知核、二次数据项、0<p<1；有限罚参数日程得到局部数值解。'),
 'blind_dcp': ('暗通道盲反卷积（Blind-DCP）','交替估计暗通道/梯度稀疏图像和非负模糊核。','deconvolution/blind.py','二维灰度或末轴多通道、周期模型；输入核支持，不提供真核。'),
 'blind_l0': ('近 L0 盲反卷积（Blind-L0）','交替估计近 L0 梯度图像和模糊核。','deconvolution/blind.py','二维灰度或末轴多通道、周期模型；阈值层数和金字塔控制实际内部预算。'),
}

def demo(name):
    output=f'outputs/algorithm_examples/{name}'
    if name in {'admm','pdhg','fista'}:
        script='examples/composition_quickstart.py'
        return script, f'python {script} --domain mri --algorithm {name} --regularizer {"tv_haar" if name=="admm" else "tikhonov"} --output {output}', '脚本 configuration() 生成固定2×2复数图像；中心正交FFT产生 kspace (1,1,2,2)，轴为批次、线圈、y、x。maps和全采样mask为1，信号采用合成例单位，无裁剪或强度拟合。', '(1,2,2) complex128 reconstruction.npy 和 result.json。'
    if name in {'landweber','cgls','lsqr','cg'}:
        script='examples/quadratic_quickstart.py'
        return script,f'python {script} --domain mri --method {name} --output {output}', '脚本生成固定4×4复数对象；单线圈maps、mask为1，中心正交FFT得到 kspace (1,1,4,4)。恒等正则权重0.08，无裁剪或强度拟合。', '(1,4,4) complex128 reconstruction.npy 和 result.json。'
    if name=='mlem':
        return 'examples/emission_quickstart.py',f'python examples/emission_quickstart.py --algorithm mlem --output {output}', '脚本生成8×8非负方块和8个[0,π)平行束角度（弧度）；以投影为均值生成固定种子Poisson计数。observation=(1,1,8,8)，轴为批次、通道、角度、探测器，单位为计数；不作负对数。', '(1,1,8,8)非负 reconstruction.npy 和 result.json；示例执行两轮。'
    if name in {'fbp','fdk','sirt','sart','asd_pocs_tv'}:
        case='cone_3d/spheres_astra_12' if name=='fdk' else 'parallel_2d__ellipses_dense_clean' if name=='fbp' else 'parallel_2d__ellipses_sparse_gaussian'
        config=f'configs/algorithms/{name}.yaml'
        data=('tests/data/cases/cone_3d__spheres_astra_12/arrays.h5：truth/x=(1,12,12,12)，measurement/y_observed=(1,24,36,24)，投影轴为批次、探测器行、角度、探测器列。case.json 给出弧度角和归一化几何距离；无噪线积分。' if name=='fdk' else f'tests/data/cases/{case}/arrays.h5：truth/x=(1,1,64,64)，measurement/y_observed=(1,1,{180 if name=="fbp" else 32},64)，投影轴为批次、通道、角度、探测器。case.json 随附弧度角及归一化距离；原图固定乘0.02。'+('使用干净线积分。' if name=='fbp' else '投影加入1% RMS高斯噪声，不作负对数。'))
        return config,f'invct run {name} --case {case} --config {config} --out {output} --device {"cuda" if name=="fdk" else "cpu"}',data,('三维(1,12,12,12)' if name=='fdk' else '二维(1,1,64,64)')+'重建；reconstruction.pt 保存tensor bundle，comparison.png可直接查看，metrics.json与diagnostics.json保存指标、参数和停止状态。'
    domain='mri' if name in {'zero_filled','grappa','espirit','varnet'} else 'deconvolution'
    case='standard_mr_cartesian' if name=='varnet' else 'standard_mr_dev' if domain=='mri' else 'public_camera_levin'
    config=f'configs/{domain}/benchmarks/standard_{case}_{"poisson" if name=="richardson_lucy" else "gaussian"}_{name}.yaml'
    if domain=='mri':
        size=256 if name=='varnet' else 128
        data=f'tests/data/mri/cases/{case}/public.npz：'
        data+=('calibration_data=(1,4,128,32)，为连续采集ACS，输出网格128×128。' if name=='espirit' else f'kspace与sampling_mask=(1,4,{size},{size})；'+('GRAPPA另读calibration_data。' if name=='grappa' else 'SENSE线圈组合另读同形sensitivity_maps。' if name=='zero_filled' else '网络从采样数据估计线圈信息。'))
        data+='数组轴为批次、线圈、y、x，采用中心正交FFT。模拟MR信号已包含组织对比度和复数噪声，运行时不再裁剪或重新归一化。'
        expected='sensitivity_maps.npy (1,4,128,128)、eigenvalues.npy和result.json，供后续重建调用。' if name=='espirit' else f'reconstruction.npy (1,{size},{size}) 和result.json；'+('GRAPPA为RSS幅值，正常输出有限。' if name=='grappa' else '重建形状与输入网格一致且有限。')
    else:
        data=f'tests/data/deconvolution/cases/{case}/public.npz：'+('counts=(256,256)，单位为光子计数；gain=40、background=0.2。' if name=='richardson_lucy' else 'gaussian=(256,256)为含噪观测。')
        data+=('盲任务仅读取观测，kernel_shape=[19,19]由配置指定。' if name.startswith('blind') else 'psf=(19,19)，原点[9,9]像素。')
        data+='camera图像已抗混叠缩放为256×256、uint8除255，采用周期卷积；图像轴为y、x，不在重建时重新归一化。'
        expected='reconstruction.npy (256,256) 和result.json；'+('另有psf.npy (19,19)。' if name.startswith('blind') else '正常输出为有限图像。')
    return config,f'python -m inv_framework.runner --config {config} --output {output}',data,expected

def domain_rules(p):
    terms=[p.get('type','数值')]
    if p.get('nullable'):terms.append('可为null')
    for key,sign in [('minimum','≥'),('maximum','≤')]:
        if p.get(key) is not None:
            if p.get('exclusive_'+key,not p.get('inclusive_'+key,True)):sign='>' if key=='minimum' else '<'
            terms.append(sign+str(p[key]))
    if p.get('choices'):terms.append(' / '.join(map(str,p['choices'])))
    if p.get('theoretical_constraint'):terms.append(p['theoretical_constraint'])
    return '；'.join(terms)

def render(only=None):
    from inv_framework.api import list_capabilities as ct
    from inv_framework.mri.api import list_capabilities as mri
    from inv_framework.deconvolution.api import list_capabilities as deconv
    groups=defaultdict(list)
    for domain,discover in [('ct',ct),('mri',mri),('deconvolution',deconv)]:
        for record in discover()['algorithms']:
            groups[Path(algorithm_documentation(record['name'])).stem].append((domain,record))
    selected=None if only is None else {Path(algorithm_documentation(n)).stem for _,n in only}
    destination=ROOT/'docs/algorithms';destination.mkdir(exist_ok=True)
    index=['# 算法使用说明','','按数值方法组织。共22个算法／方法家族，映射46个公开任务入口；同一方法的领域适配、正则和保真项配置放在同一页。OS-SART、OSEM、RL-TV作为对应家族的更新变体保留说明。','','| 算法／家族 | 所属任务 | 公开入口 |','| --- | --- | --- |']
    for family,entries in groups.items():
        title,purpose,module,scope=METHODS[family]
        tasks='、'.join(dict.fromkeys(DOMAINS[d] for d,_ in entries))
        ids='；'.join(f"{DOMAINS[d]}: `{r['name']}`" for d,r in entries)
        index.append(f'| [{title}]({family}.md) | {tasks} | {ids} |')
        if selected is not None and family not in selected:continue
        config,command,data,expected=demo(family)
        lines=[f'# {title}','','## 一、算法身份','',f'- **名称**：{title}。',f'- **一句话简介**：{purpose}',f'- **所属任务**：{tasks}。',f'- **方法类型**：{"学习类，使用训练权重" if family=="varnet" else "非学习类，无需训练权重"}。',f'- **适用范围**：{scope}',f'- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/{module}](../../inv_framework/{module})。','', '### 领域入口与目标','', '| 领域／入口 | 适用范围与实际目标 |','| --- | --- |']
        for d,r in entries:
            objective=r.get('objective_definition',r.get('objective',''))
            description=(r.get('applicability',r.get('description',''))+' '+str(objective)).replace('|','\\|')
            lines.append(f"| {DOMAINS[d]} / `{r['name']}` | {description} |")
        if family=='varnet':lines+=['','网络组件改编自 [facebookresearch/fastMRI](https://github.com/facebookresearch/fastMRI) archived main（仓库记录获取日期2026-11-06）；MIT版权声明见[第三方来源](../../THIRD_PARTY_NOTICES.md)。']
        if family=='fdk':lines+=['','后端采用 [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox) 的 FDK_CUDA；ASTRA按其GPL许可独立安装。']
        if family in {'admm','pdhg','fista','landweber','cgls','lsqr','cg'}:lines+=['','共享入口支持二维平行束CT、二维Cartesian SENSE及已知PSF二维反卷积。专用入口的维度、边界和正则约定保留各自语义；MRI使用原生复数和Hermitian伴随。共同线性基线只接收二次目标。']
        lines+=['','## 二、输入输出','','### 最小示例数据','',data,'',f'示例配置或生成脚本：[{config}](../../{config})。','',f'**正常输出**：{expected}','', 'NPY用`numpy.load`读取；tensor bundle用`torch.load(..., weights_only=True)`读取。结果同时记录执行状态、实际迭代和停止原因；预算结束与数值收敛分别解释。','', '### 其他领域入口','']
        for d in dict.fromkeys(d for d,_ in entries):lines.append(f'- {DOMAINS[d]}：[{d}输入契约](../{d}/data_format.md)，说明数组轴、几何/采样、单位及预处理。')
        lines+=['','## 三、运行方式','','从算法库根目录执行。Python ≥3.10；基础依赖为PyTorch、NumPy、SciPy、h5py、PyYAML、matplotlib：','','```bash','python -m pip install -e ".[dev]"','```','', ('本例使用CUDA，需支持CUDA的PyTorch和ASTRA。' if family=='fdk' else '本例使用CPU。')+'完整安装及Windows Conda设置见[依赖说明](../../DEPENDENCIES.md)。无需Agent SDK或大模型服务。']
        if family=='varnet':lines+=['','配置读取`outputs/weights/knee_leaderboard_state_dict.pt`。该模型文件独立于pip安装；从[fastMRI官方knee权重](https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/knee_leaderboard_state_dict.pt)取得后放置于此，或修改checkpoint路径。级联数、通道数、池化深度须匹配，示例ACS宽度为48。具体加载约定见[MRI模型说明](../mri/algorithms.md)。']
        lines+=['','### 最小示例命令','','```bash',command,'```','', '每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。']
        if family in {'admm','pdhg','fista','landweber','cgls','lsqr','cg'}:lines+=['','脚本的`--domain`可选ct、mri、deconvolution。线性CT示例读取64×64椭圆catalog，组合CT示例生成2×2匹配投影；MRI保留非平凡相位，反卷积使用已知小PSF。完整数据起点见[组合示例](../../examples/composition/README.md)。']
        if family=='mlem':lines+=['','将`--algorithm mlem`改为`--algorithm osem`运行两子集变体。']
        lines+=['','## 四、参数','','下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。','', '建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。']
        for d,r in entries:
            n=r['name'];source=f'configs/{"" if d=="ct" else d+"/"}algorithms/{n}.yaml'
            defaults=r['default_parameters'] if d=='ct' else r['parameters']
            params=r['parameter_specs'] if d=='ct' else [dict(name=k,**v) for k,v in r['parameter_schema'].items()]
            lines+=['',f'### {DOMAINS[d]}：`{n}`','',f'默认值来源：[本仓库配置](../../{source})。'+('未在YAML显式填写的值由[CT注册表](../../inv_framework/ct/registry.py)补齐。' if d=='ct' else ''),'','| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |','| --- | --- | --- | --- | --- | --- |']
            for p in params:
                search=p.get('recommended_search',{})
                recommended='固定数值/采集条件，按任务显式覆盖' if search.get('sampling')=='fixed' else str(search.get('values',[]))
                values=[p['name'],PARAMETER_HELP.get(p['name'],p.get('description',p['name']))+'；'+p.get('units',''),defaults.get(p['name']),f'`{source}`',domain_rules(p),recommended]
                lines.append('| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in values)+' |')
            f=r['fidelity'];lines+=['',f"保真项模式：`{f['mode']}`；可执行项：`{f['supported']}`。{f['reason']}"]
        coupling={
            'fbp':'scale为几何尺度；不通过重建后强度拟合改变指标。',
            'fdk':'filter_type与short_scan须匹配扫描范围；voxel_supersampling为正整数。',
            'sart':'subset_count、block_size须与实际视角数形成合法划分；固定seed保证相同调度。',
            'mlem':'计数非负、初值及数值除法下限为正；OSEM子集须匹配视角。',
            'sirt':'上下界须有序，迭代预算是上限，不保证恰好执行到上限。',
            'asd_pocs_tv':'epsilon与残差度量必须匹配，epsilon_source说明半径来源；TV下降和投影预算分别记录。',
            'zero_filled':'SENSE需要maps，RSS参考必须采用RSS定义；不存在优化权重搜索。',
            'grappa':'kernel_width为奇数并适配ACS支持，ridge采用实际校准矩阵的相对缩放。',
            'espirit':'kernel_width为奇数并适配ACS；0<svd_threshold<1，0≤crop≤1。',
            'varnet':'网络结构与checkpoint严格匹配；num_low_frequencies对应实际采集的ACS列宽。',
            'wiener_hunt':'balance为非负Laplacian平方项系数；需周期已知核，gain=1、background=0。',
            'richardson_lucy':'Poisson均值包含gain和background；RL-TV的平滑TV分母须保持有效。',
            'hyper_laplacian':'0<p<1，beta_factor>1且beta_max≥beta_initial；比较时固定精度与罚参数日程。',
            'blind_dcp':'核支持为图像内奇数尺寸且至少3；patch_size为奇数，penalty_factor>1。',
            'blind_l0':'核支持为图像内奇数尺寸且至少3；阈值逐层减半，内部预算随层数增加。',
        }.get(family,'权重对应未按像素数平均的目标。Haar层数须适配尺寸；rho为正且与正则权重独立；步长和CG精度依据实际算子与系统。MRI复数变量不接受实数盒约束，具体组合按入口条件执行。')
        lines+=['','### 耦合条件','',coupling,'','目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。']
        (destination/f'{family}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if selected is None:
        (destination/'README.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
        for entries in groups.values():
            for d,r in entries:
                old=destination/f"{d}_{r['name']}.md"
                if old.exists():old.unlink()


if __name__=='__main__':render()
