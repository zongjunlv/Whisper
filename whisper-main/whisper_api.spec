# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['whisper_api.py'],
    pathex=[],
    binaries=[],
    datas=[
        (r"E:\Proj_JiangZhuYing\whisper-main\weights", "./weights"),
        (r"E:\Proj_JiangZhuYing\whisper-main\whisper", "./whisper"),
        (r"C:\Users\17655\anaconda3\envs\whisper\Lib\site-packages\zhconv", "./zhconv"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='whisper_api',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='whisper-api',
)
