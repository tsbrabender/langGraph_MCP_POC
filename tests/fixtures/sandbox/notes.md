# Project Notes

## Overview

This sandbox directory is used exclusively for MCP tool unit tests.

## Files

- `hello.txt` — plain text fixture
- `data.json` — JSON fixture
- `notes.md` — Markdown fixture (this file)
- `subdir/nested.txt` — nested file for recursive search tests

## Usage

All paths passed to MCP tools are resolved relative to this directory.
Path traversal attempts (e.g. `../outside`) are blocked by the sandbox enforcer.
