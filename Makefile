FOLDER_PROJECT = src/cartola

.PHONY: clean
clean:
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' | xargs rm -rf
	@find . -type d -name '*.egg*' | xargs rm -rf
	@find . -type d -name '*.ropeproject' | xargs rm -rf
	@rm -rf src/build/
	@rm -rf src/dist/
	@rm -rf docs/build/
	@rm -rf references/
	@rm -rf results/
	@rm -f MANIFEST
	@rm -f .coverage.*

ruff:
	@uv run --frozen scripts/pyrepo-check ruff

ty:
	@uv run --frozen scripts/pyrepo-check ty

bandit:
	@uv run --frozen scripts/pyrepo-check bandit

quality:
	@uv run --frozen scripts/pyrepo-check --all

pre-commit:
	@pre-commit run --all-files

docker-build:
	@kedro docker build --image cartola

docker-run:	
	@kedro docker run --image cartola

docker:
	$(MAKE) docker-build
	$(MAKE) docker-run
