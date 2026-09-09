# Empty on purpose. pytest inserts this file's directory (the repo
# root) into sys.path when it exists, which is what lets tests import
# top-level scripts like server.py (added in a later task) with a plain
# `from server import ...` instead of packaging them.
