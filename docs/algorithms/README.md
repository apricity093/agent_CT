# 算法使用说明

按数值方法组织。共22个算法／方法家族，映射46个公开任务入口；同一方法的领域适配、正则和保真项配置放在同一页。OS-SART、OSEM、RL-TV作为对应家族的更新变体保留说明。

| 算法／家族 | 所属任务 | 公开入口 |
| --- | --- | --- |
| [滤波反投影 FBP](fbp.md) | CT | CT: `fbp` |
| [SIRT](sirt.md) | CT | CT: `sirt` |
| [Landweber](landweber.md) | CT、MRI、反卷积 | CT: `landweber`；MRI: `landweber`；反卷积: `landweber` |
| [CGLS](cgls.md) | CT、MRI、反卷积 | CT: `cgls`；MRI: `cgls`；反卷积: `cgls` |
| [LSQR](lsqr.md) | CT、MRI、反卷积 | CT: `lsqr`；MRI: `lsqr`；反卷积: `lsqr` |
| [共轭梯度 CG](cg.md) | CT、MRI、反卷积 | CT: `cg`；CT: `tikhonov`；MRI: `cg`；MRI: `cg_sense`；反卷积: `cg` |
| [SART](sart.md) | CT | CT: `sart`；CT: `os_sart` |
| [最大似然 EM（MLEM）](mlem.md) | CT | CT: `mlem`；CT: `osem` |
| [FISTA](fista.md) | CT、MRI、反卷积 | CT: `tv_fista`；CT: `fista`；MRI: `fista`；反卷积: `fista` |
| [ADMM](admm.md) | CT、MRI、反卷积 | CT: `admm`；MRI: `admm`；MRI: `cs_tv`；MRI: `cs_wavelet`；反卷积: `admm`；反卷积: `tv_admm` |
| [PDHG](pdhg.md) | CT、MRI、反卷积 | CT: `pdhg`；CT: `constrained_pdhg`；MRI: `pdhg`；反卷积: `pdhg` |
| [ASD-POCS-TV](asd_pocs_tv.md) | CT | CT: `asd_pocs_tv` |
| [锥束滤波反投影 FDK](fdk.md) | CT | CT: `fdk` |
| [ESPIRiT](espirit.md) | MRI | MRI: `espirit` |
| [GRAPPA](grappa.md) | MRI | MRI: `grappa` |
| [VarNet](varnet.md) | MRI | MRI: `varnet` |
| [Zero-filled](zero_filled.md) | MRI | MRI: `zero_filled` |
| [暗通道盲反卷积（Blind-DCP）](blind_dcp.md) | 反卷积 | 反卷积: `blind_dcp` |
| [近 L0 盲反卷积（Blind-L0）](blind_l0.md) | 反卷积 | 反卷积: `blind_l0` |
| [Hyper-Laplacian 反卷积](hyper_laplacian.md) | 反卷积 | 反卷积: `hyper_laplacian` |
| [Richardson–Lucy](richardson_lucy.md) | 反卷积 | 反卷积: `richardson_lucy`；反卷积: `rl_tv` |
| [Wiener–Hunt](wiener_hunt.md) | 反卷积 | 反卷积: `wiener_hunt` |
