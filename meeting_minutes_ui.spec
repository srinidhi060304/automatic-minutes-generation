# -*- mode: python ; coding: utf-8 -*-

import os
import logging
from PyInstaller.utils.hooks import collect_data_files

# Configure logging for errors and warnings only
logging.basicConfig(
    filename=os.path.join(r"C:\Users\admin\Documents\drdo", 'build_log.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PyInstaller')

block_cipher = None

# Define base directory
base_dir = r"C:\Users\admin\Documents\drdo"

# Collect .whl files from packages folder
whl_files = [(os.path.join(base_dir, "packages", f), "packages") for f in os.listdir(os.path.join(base_dir, "packages")) if f.endswith(".whl")]
if not whl_files:
    logger.error("No .whl files found in %s", os.path.join(base_dir, "packages"))
    raise FileNotFoundError("No .whl files found in packages folder")

# Collect additional files (binaries, models, and MP3)
bin_files = [
    (os.path.join(base_dir, "bin", "ffmpeg.exe"), "bin"),
    (os.path.join(base_dir, "bin", "ffplay.exe"), "bin"),
    (os.path.join(base_dir, "bin", "ffprobe.exe"), "bin")
]

model_files = [
    (os.path.join(base_dir, "models", "bart-samsum"), "models/bart-samsum"),
    (os.path.join(base_dir, "models", "pyannote"), "models/pyannote"),
    (os.path.join(base_dir, "models", "small"), "models/small"),
    (os.path.join(base_dir, "models", "speechbrain"), "models/speechbrain")
]

additional_files = [
    (os.path.join(base_dir, "002145_a-conversation-with-a-neighbor-53032.mp3"), ".")
]

# Combine all data files
datas = whl_files + bin_files + model_files + additional_files

# List of packages from requirements.txt (for hiddenimports)
hiddenimports = [
    'pyannote.audio',
    'pyannote.core',
    'pyannote.database',
    'pyannote.metrics',
    'pyannote.pipeline',
    'faster_whisper',
    'ctranslate2',
    'speechbrain',
    'speechbrain.pretrained',
    'torch',
    'torchaudio',
    'pytorch_lightning',
    'torchmetrics',
    'transformers',
    'sklearn',
    'pkg_resources',
    'numpy',
    'scipy',
    'pydub',
    'PyQt5',
    'matplotlib',
    'pandas',
    'optuna',
    'lightning',
    'accelerate',
    'aiohappyeyeballs',
    'aiohttp',
    'aiosignal',
    'alembic',
    'altgraph',
    'antlr4_python3_runtime',
    'asteroid_filterbanks',
    'attrs',
    'audioread',
    'av',
    'bitsandbytes',
    'certifi',
    'cffi',
    'charset_normalizer',
    'click',
    'colorama',
    'coloredlogs',
    'colorlog',
    'contourpy',
    'cycler',
    'decorator',
    'docopt',
    'einops',
    'ffmpeg_python',
    'filelock',
    'flatbuffers',
    'fonttools',
    'frozenlist',
    'fsspec',
    'future',
    'greenlet',
    'hf_xet',
    'huggingface_hub',
    'humanfriendly',
    'HyperPyYAML',
    'idna',
    'intel_openmp',
    'Jinja2',
    'joblib',
    'julius',
    'kiwisolver',
    'lazy_loader',
    'librosa',
    'lightning_utilities',
    'llvmlite',
    'Mako',
    'markdown_it_py',
    'MarkupSafe',
    'mdurl',
    'mkl',
    'mpmath',
    'msgpack',
    'multidict',
    'networkx',
    'numba',
    'omegaconf',
    'onnxruntime',
    'packaging',
    'pefile',
    'Pillow',
    'platformdirs',
    'pooch',
    'primePy',
    'propcache',
    'protobuf',
    'psutil',
    'pycparser',
    'Pygments',
    'pyinstaller',
    'pyinstaller_hooks_contrib',
    'pyparsing',
    'PyQt5_Qt5',
    'PyQt5_sip',
    'pyreadline3',
    'python_dateutil',
    'pytorch_metric_learning',
    'pytz',
    'pywin32_ctypes',
    'PyYAML',
    'regex',
    'requests',
    'rich',
    'ruamel_yaml',
    'ruamel_yaml_clib',
    'safetensors',
    'scikit_learn',
    'scipy',
    'semver',
    'sentencepiece',
    'setuptools',
    'shellingham',
    'six',
    'sortedcontainers',
    'sounddevice',
    'soundfile',
    'soxr',
    'SQLAlchemy',
    'sympy',
    'tabulate',
    'tbb',
    'tensorboardX',
    'threadpoolctl',
    'tokenizers',
    'torch_audiomentations',
    'torch_pitch_shift',
    'torchvision',
    'tqdm',
    'typer',
    'typing_extensions',
    'tzdata',
    'urllib3',
    'yarl'
]

# Collect data files for specific packages
try:
    datas += collect_data_files('pyannote.audio')
    datas += collect_data_files('speechbrain')
    datas += collect_data_files('faster_whisper')
    datas += collect_data_files('transformers')
    datas += collect_data_files('torch')
    datas += collect_data_files('torchaudio')
    datas += collect_data_files('matplotlib')
    datas += collect_data_files('pandas')
    datas += collect_data_files('optuna')
except Exception as e:
    logger.error("Error collecting data files: %s", str(e))

# PyInstaller analysis
try:
    a = Analysis(
        ['meeting_minutes_ui.py', 'test2.py'],
        pathex=[os.path.join(base_dir, "packages")],
        binaries=[],
        datas=datas,
        hiddenimports=hiddenimports,
        hookspath=[],
        hooksconfig={},
        runtime_hooks=[],
        excludes=['onnxruntime.gpu', 'cuda', 'cudnn', 'bitsandbytes.cuda', 'torch.cuda', 'pysqlite2', 'MySQLdb'],
        win_no_prefer_redirects=False,
        win_private_assemblies=False,
        cipher=block_cipher,
        noarchive=True,
    )
except Exception as e:
    logger.error("Error during Analysis: %s", str(e))
    raise

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

try:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='MeetingMinutesGenerator',
        debug=True,  # Enable debug mode for detailed output
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
except Exception as e:
    logger.error("Error creating EXE: %s", str(e))
    raise

try:
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='MeetingMinutesGenerator',
    )
except Exception as e:
    logger.error("Error during COLLECT: %s", str(e))
    raise