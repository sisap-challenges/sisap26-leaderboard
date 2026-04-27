import h5py
import os
from urllib.request import urlretrieve
from pathlib import Path
from scipy.sparse import csr_matrix

def download(src, dst):
    print(dst)
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def load_sparse_matrix(h5_group):
    """Reconstructs a SciPy CSR matrix from HDF5 datasets."""
    indptr = h5_group['indptr'][:]
    indices = h5_group['indices'][:]
    data = h5_group['data'][:]
    shape = tuple(h5_group.attrs['shape'])
    return csr_matrix((data, indices, indptr), shape=shape)

def get_fn(dataset, task):
    cfg = DATASETS[dataset][task]
    if 'local_fn' in cfg:
        return cfg['local_fn']
    return os.path.join("data", dataset, task, f"{dataset}.h5")

def get_gt_fn(dataset, task):
    """Return the path to the ground-truth file.
    Falls back to get_fn() when the dataset has no separate GT file."""
    cfg = DATASETS[dataset][task]
    if 'gt_local_fn' in cfg:
        return cfg['gt_local_fn'](dataset, task)
    return get_fn(dataset, task)

def prepare(dataset, task):
    cfg = DATASETS[dataset][task]
    if cfg.get('url') is not None:
        download(cfg['url'], get_fn(dataset, task))
    # Download separate ground-truth file if the dataset defines one
    if 'gt_url' in cfg and cfg['gt_url'] is not None:
        download(cfg['gt_url'], get_gt_fn(dataset, task))

def get_query_count(dataset, task):
    fn = get_fn(dataset, task) 
    f = h5py.File(fn)
    qn = len(DATASETS[dataset][task]['queries'](f))
    f.close()
    return qn

SISAP25_BASE_URL = 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/'

DATASETS = {
    # ------------------------------------------------------------------ #
    # SISAP 2025 – Task 2: all-kNN graph on GOOAQ (3 M vectors, 384-dim) #
    # ------------------------------------------------------------------ #
    'gooaq': {
        'task1': {
            # Main dataset file.  The on-disk layout of the 'train' array
            # is verified at runtime by load_real_vectors() in
            # create-gooaq-small.py; pass it through as-is here and let
            # FAISS normalise it during search.
            'url': SISAP25_BASE_URL + 'benchmark-dev-gooaq.h5',
            # Ground truth lives in a separate file.
            # knns shape: (N, 32) – row-major, 1-indexed, self-loop included.
            'gt_url': SISAP25_BASE_URL + 'allknn-benchmark-dev-gooaq.h5',
            'gt_local_fn': lambda dataset, task: os.path.join(
                "data", dataset, task, "allknn-gooaq.h5"
            ),
            'data': lambda x: x['train'],
            'gt_I': lambda x: x['knns'],   # shape (N, 32), 1-indexed
            'k': 15,
        }
    },
    # ------------------------------------------------------------------ #
    # Small synthetic example (10 K vectors) for quick tests / sharing.  #
    # Generate with:  python create-gooaq-small.py                        #
    # ------------------------------------------------------------------ #
    'gooaq-small': {
        'task1': {
            # No remote URL – file is created locally by create-gooaq-small.py
            'url': None,
            'local_fn': 'data/benchmark-dev-gooaq-small.h5',
            'data': lambda x: x['train'],
            'gt_I': lambda x: x['allknn']['knns'],   # shape (N, 32), 1-indexed
            'k': 15,
        }
    },
    'wikipedia-small': {
        'task1': {
            'url': 'https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/main/benchmark-dev-wikipedia-bge-m3-small.h5',
            'data': lambda x: x['train'],
            'gt_I': lambda x: x['allknn']['knns'],
            'k': 15,
        }
    },
    'llama-dev': {
        'task2' : {
            'url': 'https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/main/llama-dev.h5',
            'queries': lambda x: x['test']['queries'],
            'data': lambda x: x['train'],
            'gt_I': lambda x: x['test']['knns'],
            'k': 30,
        }
    },
    'nq': {
        'task3': {
            'url': 'https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/refs%2Fpr%2F3/nq.h5',
            'queries': lambda x: load_sparse_matrix(x['otest']['queries']),
            'data': lambda x: load_sparse_matrix(x['train']),
            'gt_I': lambda x: x['otest']['knns'],
            'k': 30,
        }
    },
    'fiqa-dev': {
        'task3': {
            'url': 'https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/refs%2Fpr%2F3/fiqa.h5',
            'queries': lambda x: load_sparse_matrix(x['otest']['queries']),
            'data': lambda x: load_sparse_matrix(x['train']),
            'gt_I': lambda x: x['otest']['knns'],
            'k': 30,            
        }
    }
}