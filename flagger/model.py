"""Only small supervised scikit-learn classifiers."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from flagger.text import normalize_text


def make_pipeline(kind: str = "word_char_logistic") -> Pipeline:
    word = TfidfVectorizer(
        preprocessor=normalize_text,
        ngram_range=(1, 2),
        min_df=1,
        max_features=18000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    if kind == "word_char_logistic":
        features = FeatureUnion([
            ("word", word),
            ("char", TfidfVectorizer(
                preprocessor=normalize_text,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=22000,
                sublinear_tf=True,
            )),
        ])
    else:
        features = word
    if kind == "word_naive_bayes":
        classifier = MultinomialNB(alpha=0.5)
    elif kind in {"word_logistic", "word_char_logistic"}:
        classifier = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {kind}")
    return Pipeline([("features", features), ("classifier", classifier)])
