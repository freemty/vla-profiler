# Deploy New Model Package to xdlab23

> 当 GitHub 被防火墙封锁时，如何将新模型的源码包部署到 xdlab23

## Problem
xdlab23 无法直接 git clone GitHub 仓库。需要在本地下载后传输。

## Cause
xdlab23 校园网防火墙封锁了 GitHub。

## Solution

### Step 1: 本地 Sparse Clone (只下载需要的文件)

```bash
git clone --filter=blob:none --sparse https://github.com/{org}/{repo}.git /tmp/{repo}
cd /tmp/{repo}
git sparse-checkout set {src_dir} setup.py pyproject.toml
```

`--filter=blob:none --sparse` 只下载目录结构，`sparse-checkout set` 按需下载文件内容。比完整 clone 快很多。

### Step 2: 打包传输

```bash
tar czf /tmp/{repo}.tar.gz -C /tmp {repo}
scp -P 66 /tmp/{repo}.tar.gz xdlab23_yang:/data1/ybyang/
```

### Step 3: 服务器端解压安装

```bash
ssh xdlab23_yang
cd /data1/ybyang
tar xzf {repo}.tar.gz
cd {repo}

# 激活 conda (非 login shell 需要手动 source)
. /home/ybyang/miniconda3/etc/profile.d/conda.sh
conda activate vit-probe

pip install -e .
```

### Step 4: 验证安装

```bash
python -c "import {package_name}; print('OK')"
```

## Alternative: Git Bundle (适合自己的 repo)

对于自己的代码（如 vlla），用 `scripts/sync_to_remote.sh` 走 git bundle：
```bash
git bundle create /tmp/repo.bundle --all
scp bundle → server → git clone bundle
```

Git bundle 保留完整 git 历史，适合需要 git 操作的场景。第三方模型库通常只需要 pip install，用 tar 更简单。

## Conda 注意事项

通过 SSH 运行远程命令时（非 login shell），conda 不会自动加载：
```bash
# 错误: conda: command not found
ssh xdlab23_yang "conda activate vit-probe && ..."

# 正确: 手动 source conda init
ssh xdlab23_yang ". /home/ybyang/miniconda3/etc/profile.d/conda.sh && conda activate vit-probe && ..."
```

`scripts/launch_exp.sh` 已内置此处理。

## Notes
- Date: 2026-04-22
- Environment: xdlab23, SSH port 66
- Tested with: NitroGen (MineDojo), lingbotvla
