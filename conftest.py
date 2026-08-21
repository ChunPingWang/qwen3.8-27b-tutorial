"""放在專案根目錄的 conftest.py 會讓 pytest 把根目錄加入 sys.path，
tests/ 底下才能直接 `from agent import ...`。"""
