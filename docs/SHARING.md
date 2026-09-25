# 与队友共享和上传 Git

分享范围是整个 MMSA-PMF 源码目录，包含 src、scripts、configs、constraints、tests、docs、README 和原始 LICENSE。队友不需要服务器上的上层项目目录即可安装、测试和运行。
数据、恢复标签、BERT 权重和训练产物单独共享。data/README.md 描述数据接口；不要把它们混进源码提交。

## 从源码快照创建新仓库

解压 MMSA-PMF-share.zip 后，在 MMSA-PMF 目录执行：

```bash
git init
git add .
git diff --cached --stat
git commit -m "Add PMF baseline on MMSA"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

YOUR_REPOSITORY_URL 替换为团队实际仓库地址。快照不含 .git；这些操作会创建新的历史。当前整理任务没有创建或推送远程仓库。

若直接使用原服务器 checkout，希望保留上游历史，先用 git remote -v 核对地址；其 origin 原来指向上游 MMSA 镜像。可将该远程重命名为 upstream，再添加团队 origin。不要把原上游地址当作团队目标。

## 首次克隆后的验证

按 docs/INSTALL.md 安装，运行 python -m unittest discover -s tests -v，再使用真实附件执行 train_pmf.py --dry-run。
数据路径通过 --data、--test2 指定，产物目录通过 --output-dir 指定；不需要修改源码内的服务器路径。

同一输出目录可顺序追加多次实验，run_id 区分各次运行。不同进程并发训练时使用不同 --output-dir，避免并发写同一 CSV 或日志。
GitHub Actions 仅运行不依赖竞赛数据的测试；不训练正式实验、不自动发布 PyPI。
