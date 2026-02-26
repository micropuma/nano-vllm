.PHONY: format check test

format:
	# 安全地自动修复，--unsafe-fixes 支持不安全修复
	ruff check --fix   
	ruff format

check:
	ruff check
	ruff format --check

test:
	python bench.py