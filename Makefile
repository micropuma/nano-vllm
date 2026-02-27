.PHONY: format check test

format:
	# 安全地自动修复，--unsafe-fixes 支持不安全修复
	ruff check --fix   
	ruff format

check:
	ruff check
	ruff format --check

bench:
	python bench.py

profile:
	nsys profile \
		-t cuda,nvtx,osrt,cublas \
		-o ./report \
		--force-overwrite=true \
		python example.py

pytorch-profile:
	python trace.py