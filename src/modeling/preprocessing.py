"""Single ColumnTransformer shared by all candidate models."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _numeric_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def _categorical_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    high_card: list[str] | None = None,
    family: str | None = None,
) -> ColumnTransformer:
    """
    Preprocessamento único para todos os modelos:
    - numéricas: mediana + StandardScaler
    - categóricas: DESCONHECIDO + One-Hot

    `high_card` e `family` existem só por compatibilidade de assinatura;
    categorias de alta cardinalidade devem ser filtradas em `feature_lists`.
    """
    del family  # mesmo prep para linear e árvore
    cats = list(categorical)
    if high_card:
        cats.extend(c for c in high_card if c not in cats)

    transformers: list[tuple] = []
    if numeric:
        transformers.append(("num", _numeric_pipeline(), numeric))
    if cats:
        transformers.append(("cat", _categorical_pipeline(), cats))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
