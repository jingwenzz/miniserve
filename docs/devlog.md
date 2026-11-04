# MiniServe Devlog

Day 1 / 2026-11-04:
今天要让系统能够: 从干净环境重复安装, 跑通现有测试和 demo, 并留下可查的版本记录。
实际改了什么: 新建本地虚拟环境 .venv (已在 .gitignore 中, 不提交); 按 pyproject.toml 以 editable 方式安装开发依赖; 新建本文件作为第一条记录; 更新 README 的 Quickstart 小节, 改为与本次实测一致的虚拟环境流程。核心代码 request.py, scheduler.py, executor.py, engine.py 本日只读未改。本地副本不是 git 仓库, 因此没有本地分支, 后续变更按计划通过 GitHub 网页编辑器提交到远程仓库。
运行命令与观察到的结果: python3 -m venv .venv 成功, 环境内 Python 为 3.12.3。命令 .venv/bin/python -m pip install -e '.[dev]' 成功, 关键版本为 pytest 9.2.0, ruff 0.17.3, mypy 2.4.1, pip 24.0, 系统为 Linux x86_64 glibc 2.39。命令 .venv/bin/python -m pytest -q 结果为 17 passed in 0.06s, 退出码 0。命令 .venv/bin/python benchmarks/demo_scheduler.py 退出码 0, 共 5 步完成 3 个请求, 输出为 request-1 prompt 10 output 4, request-2 prompt 40 output 2, request-3 prompt 5 output 3, 与各自 max_new_tokens 一致。
我现在能解释的一件事: Engine.step 每轮先由 scheduler 选请求, 再执行 prefill 和 decode, 最后更新队列。
仍然不确定的地方: 超过 token 预算的超长 prompt 目前会一直留在 waiting 中无法执行, 这是计划中已知问题, 按计划留到 Day 3 处理, 本日不提前修。
下一次只做的一步: Day 2 定义请求与调度契约, 写 docs/design.md 的状态与队列契约。
相关 commit / run_id: 见本次提交, commit message 为 Day 1: 建立可复现的起点。若 GitHub 提交受阻, 本地成果保留并在进度文件中标注待补。
