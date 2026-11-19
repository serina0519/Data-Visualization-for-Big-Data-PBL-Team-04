"""Input validation and strictly past-fitted prompt representations."""
from pathlib import Path
import hashlib, json, zipfile, unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def load_data(path, out):
    """Retain every valid row, including missing prompts and repeated exposures.
    Missing timestamps/invalid labels fail closed rather than inventing time/labels.
    Source is auditing metadata ONLY; it is never an input feature.
    """
    path, out = Path(path), Path(out)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            candidates = [n for n in z.namelist() if n.endswith('_train.csv')]
            if len(candidates) != 1: raise ValueError('Expected exactly one *_train.csv')
            d = pd.read_csv(z.open(candidates[0]))
    else: d = pd.read_csv(path)
    required = {'uid', 'prompt', 'source', 'timestamp', 'label'}
    if not required.issubset(d): raise ValueError(f'Missing fields: {required-set(d)}')
    profile = [{'column': c, 'dtype': str(d[c].dtype), 'missing': int(d[c].isna().sum()),
                'unique': int(d[c].nunique())} for c in d]
    pd.DataFrame(profile).to_csv(out/'column_profile.csv', index=False)
    if d.uid.isna().any() or d.uid.duplicated().any(): raise ValueError('Invalid uid')
    if not d.label.isin([0, 1]).all(): raise ValueError('Invalid label; no imputation')
    d['timestamp'] = pd.to_datetime(d.timestamp, utc=True, format='mixed', errors='coerce')
    if d.timestamp.isna().any(): raise ValueError('Unknown timestamps; cannot order safely')
    d['missing_prompt'] = d.prompt.isna() | d.prompt.fillna('').str.strip().eq('')
    d['text'] = d.prompt.fillna('').map(lambda s: unicodedata.normalize('NFC', str(s)))
    d.loc[d.missing_prompt, 'text'] = '[EMPTY_PROMPT]'
    d['char_length'] = d.prompt.fillna('').str.len()
    d = d.sort_values(['timestamp', 'uid'], kind='stable').reset_index(drop=True)
    origin = d.timestamp.min().floor('D')
    d['week'] = ((d.timestamp-origin).dt.total_seconds()//(7*86400)).astype(int)+1
    d['family'] = d.source.fillna('unknown').str.split('/').str[0]
    # Canonicalisation is deliberately conservative; these are exact, not semantic, duplicates.
    d['text_hash'] = d.text.map(lambda x: hashlib.sha256(' '.join(x.split()).encode()).hexdigest())
    d['first_seen_week'] = d.groupby('text_hash').week.transform('min')
    d['novel_week'] = d.week.eq(d.first_seen_week)
    d['first_occurrence'] = ~d.text_hash.duplicated()
    initial = d.week.le(4)
    if d.loc[initial, 'label'].nunique() != 2: raise ValueError('Initial data must contain both classes')
    q1, q3 = d.loc[initial, 'char_length'].quantile([.25, .75])
    fence = float(q3+1.5*(q3-q1))
    d['long_outlier'] = d.char_length.gt(fence)  # Retain genuine extremes.
    # Length bins fixed from the initial period, not optimized on future labels.
    edges = sorted(set(d.loc[initial, 'char_length'].quantile([.25,.5,.75]).tolist()))
    d['length_bin'] = np.searchsorted(edges, d.char_length, side='right')
    weekly = d.groupby('week').agg(n=('uid','size'), positives=('label','sum'),
        missing_prompts=('missing_prompt','sum'), first_occurrences=('first_occurrence','sum'),
        time_start=('timestamp','min'), time_end=('timestamp','max')).reindex(range(1,int(d.week.max())+1))
    weekly['negatives'] = weekly.n-weekly.positives
    weekly.to_csv(out/'weekly_counts.csv')
    pd.crosstab(d.source, d.label).to_csv(out/'source_label_counts.csv')
    pd.crosstab(d.week, d.family).to_csv(out/'weekly_source_counts.csv')
    conflicts = int((d.groupby('text_hash').label.nunique()>1).sum())
    qc = dict(rows=len(d), time_start=str(d.timestamp.min()), time_end=str(d.timestamp.max()),
        origin_utc=str(origin), calendar_weeks=int(d.week.max()), observed_weeks=int(d.week.nunique()),
        initial_rows=int(initial.sum()), evaluation_rows=int((~initial).sum()), initial_weeks=4,
        missing_prompts=int(d.missing_prompt.sum()), repeated_text_rows=int(d.text_hash.duplicated().sum()),
        conflicting_text_groups=conflicts, long_outlier_fence_initial=fence,
        long_outliers_retained=int(d.long_outlier.sum()), positives=int(d.label.sum()),
        input_sha256=digest(path), length_summary=d.char_length.describe(percentiles=[.01,.05,.5,.95,.99]).to_dict(),
        scheduled_updates={str(p): int((d.week.max()-4)//p) for p in [1,2,4]},
        incomplete_tail_policy='evaluate all rows; do not update on incomplete cycles',
        exclusions=0, missing_prompt_policy='retain with fixed sentinel')
    (out/'qc.json').write_text(json.dumps(qc, indent=2, ensure_ascii=False))
    return d, qc


def embed(d, out, dim=96, local_encoder=None, kind='word'):
    """Default: frozen TF-IDF -> latent semantic embedding -> L2 normalization.
    Vocabulary, IDF and SVD are fit on weeks 1-4 ONLY. It is not a pretrained
    semantic encoder. Optional local SentenceTransformer is frozen, with no download.
    """
    out = Path(out)
    if local_encoder:
        from sentence_transformers import SentenceTransformer
        if not Path(local_encoder).is_dir(): raise ValueError('Local encoder directory required')
        model = SentenceTransformer(str(local_encoder), local_files_only=True, device='cpu')
        X = model.encode(d.text.tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        metadata = dict(type='local_sentence_transformer', path=str(local_encoder),
            max_seq_length=model.max_seq_length, note='Encoder provenance must be audited for temporal contamination')
    else:
        if kind == 'char':   # robustness variant: character 3-5-grams, same frozen protocol
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=50000,
                sublinear_tf=True, norm='l2', dtype=np.float64)
        else:
            vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=25000,
                sublinear_tf=True, norm='l2', dtype=np.float64)
        initial = d.week.le(4).values
        Z0 = vec.fit_transform(d.loc[initial,'text'])
        n_components = min(dim, Z0.shape[0]-1, Z0.shape[1]-1)
        svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=1729)
        svd.fit(Z0)
        Z = vec.transform(d.text)
        X = normalize(svd.transform(Z))
        import joblib
        joblib.dump({'vectorizer':vec, 'svd':svd}, out/'frozen_embedding.joblib')
        metadata = dict(type=f'initial_only_tfidf_svd_{kind}', dimension=n_components,
            initial_explained_variance=float(svd.explained_variance_ratio_.sum()),
            vocabulary_size=len(vec.vocabulary_), zero_vectors=int((np.linalg.norm(X,axis=1)==0).sum()),
            no_text_truncation=True, fit_weeks=[1,4])
    X = np.asarray(X, dtype=np.float64)
    np.save(out/'embeddings.npy', X)
    (out/'embedding.json').write_text(json.dumps(metadata, indent=2))
    return X
