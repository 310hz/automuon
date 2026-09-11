set dotenv-load := true
set shell := ["bash", "-uc"]

default:
    @just --list

ruff:
    uv run ruff check .
    uv run ruff format --check .

ty:
    uv run ty check

test:
    uv run pytest

lock:
    uv lock --check

check: ruff ty test lock

clean:
    rm -rf dist

build: clean
    uv build

# Verify that HEAD is clean and has a release tag.
release-check:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
        echo "error: the working tree is not clean" >&2
        exit 1
    fi
    tag="$(git describe --tags --exact-match --match 'v*' 2>/dev/null)" || {
        echo "error: HEAD must have a release tag such as v0.1.0" >&2
        exit 1
    }
    if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([ab]|rc)?[0-9]*$ ]]; then
        echo "error: unsupported release tag: $tag" >&2
        exit 1
    fi
    echo "release tag: $tag"

# Run checks and build artifacts matching the release tag on HEAD.
release-build: check release-check clean
    #!/usr/bin/env bash
    set -euo pipefail
    uv build
    tag="$(git describe --tags --exact-match --match 'v*')"
    version="${tag#v}"
    wheel="dist/automuon-${version}-py3-none-any.whl"
    sdist="dist/automuon-${version}.tar.gz"
    if [[ ! -f "$wheel" || ! -f "$sdist" ]]; then
        echo "error: built artifacts do not match release tag $tag" >&2
        exit 1
    fi

# Publish the tagged release to TestPyPI.
publish-test: release-build
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${TEST_PYPI_TOKEN:-}" ]]; then
        echo "error: TEST_PYPI_TOKEN is not set in .env" >&2
        exit 1
    fi
    UV_PUBLISH_TOKEN="$TEST_PYPI_TOKEN" uv publish \
        --publish-url https://test.pypi.org/legacy/ \
        --check-url https://test.pypi.org/simple/automuon/ \
        dist/*

# Publish the tagged release to PyPI.
publish: release-build
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${PYPI_TOKEN:-}" ]]; then
        echo "error: PYPI_TOKEN is not set in .env" >&2
        exit 1
    fi
    UV_PUBLISH_TOKEN="$PYPI_TOKEN" uv publish \
        --check-url https://pypi.org/simple/automuon/ \
        dist/*
