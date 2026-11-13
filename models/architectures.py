"""Registro e fábricas das arquiteturas usadas pelo NIDS."""

from __future__ import annotations

from typing import Any, Callable


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "cnn": {
        "kind": "supervised",
        "artifact": "cnn_model.h5",
        "description": "CNN 1D residual para classificação de flows",
    },
    "lstm": {
        "kind": "supervised",
        "artifact": "lstm/lstm_model.h5",
        "description": "LSTM para dependências sequenciais dos atributos",
    },
    "transformer": {
        "kind": "supervised",
        "artifact": "transformer/transformer_model.h5",
        "description": "Transformer tabular",
    },
    "xgboost": {
        "kind": "supervised",
        "artifact": "gradient_boosting/xgboost.joblib",
        "factory": "build_xgboost_classifier",
        "description": "Gradient Boosting otimizado para dados tabulares",
    },
    "lightgbm": {
        "kind": "supervised",
        "artifact": "gradient_boosting/lightgbm.joblib",
        "factory": "build_lightgbm_classifier",
        "description": "Gradient Boosting eficiente para alta cardinalidade",
    },
    "autoencoder": {
        "kind": "anomaly",
        "artifact": "autoencoder/autoencoder.h5",
        "description": "Autoencoder denso treinado apenas com tráfego benigno",
    },
    "vae": {
        "kind": "anomaly",
        "artifact": "vae/vae.h5",
        "description": "Variational Autoencoder para detecção de zero-day",
    },
    "isolation_forest": {
        "kind": "anomaly",
        "artifact": "isolation_forest/isolation_forest.joblib",
        "description": "Detector de outliers baseado em árvores",
    },
}


def build_xgboost_classifier(**kwargs: Any) -> Any:
    """Cria um classificador XGBoost com parâmetros conservadores para flows."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost não está instalado. Execute: pip install xgboost"
        ) from exc

    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
    }
    params.update(kwargs)
    return XGBClassifier(**params)


def build_lightgbm_classifier(**kwargs: Any) -> Any:
    """Cria um classificador LightGBM para classificação binária."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "LightGBM não está instalado. Execute: pip install lightgbm"
        ) from exc

    params = {
        "n_estimators": 300,
        "num_leaves": 63,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "binary",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update(kwargs)
    return LGBMClassifier(**params)


def build_gradient_boosting_models() -> dict[str, Any]:
    """Retorna os boosters disponíveis, informando claramente dependências ausentes."""
    models: dict[str, Any] = {}
    for name, factory in (
        ("xgboost", build_xgboost_classifier),
        ("lightgbm", build_lightgbm_classifier),
    ):
        try:
            models[name] = factory()
        except ImportError as exc:
            print(f"⚠️ Modelo {name} indisponível: {exc}")
    return models


try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:  # Permite consultar o registro sem TensorFlow instalado.
    tf = None
    keras = None
    layers = None


if layers is not None:
    class Sampling(layers.Layer):
        """Amostragem reparametrizada do espaço latente do VAE."""

        def call(self, inputs: tuple[Any, Any]) -> Any:
            z_mean, z_log_var = inputs
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        def get_config(self) -> dict[str, Any]:
            return super().get_config()


    def build_variational_autoencoder(input_dim: int, latent_dim: int = 16) -> Any:
        """Cria um VAE compacto para reconstrução de flows benignos."""
        inputs = keras.Input(shape=(input_dim,), name="flow_features")
        encoded = layers.Dense(64, activation="relu")(inputs)
        encoded = layers.Dense(32, activation="relu")(encoded)
        z_mean = layers.Dense(latent_dim, name="z_mean")(encoded)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoded)
        z = Sampling()((z_mean, z_log_var))
        decoded = layers.Dense(32, activation="relu")(z)
        decoded = layers.Dense(64, activation="relu")(decoded)
        outputs = layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(decoded)

        vae = keras.Model(inputs, outputs, name="variational_autoencoder")
        if hasattr(keras, "ops"):
            ops = keras.ops
            reconstruction_loss = ops.mean(ops.square(inputs - outputs), axis=1)
            kl_loss = -0.5 * ops.sum(
                1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1
            )
            vae.add_loss(ops.mean(reconstruction_loss + 0.001 * kl_loss))
        else:
            reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs), axis=1)
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            )
            vae.add_loss(tf.reduce_mean(reconstruction_loss + 0.001 * kl_loss))
        vae.compile(optimizer="adam")
        return vae
else:
    Sampling = None

    def build_variational_autoencoder(input_dim: int, latent_dim: int = 16) -> Any:
        """Falha explicitamente quando a dependência de deep learning está ausente."""
        raise ImportError("TensorFlow é necessário para criar o VAE.")


def get_model_factory(name: str) -> Callable[..., Any]:
    """Obtém uma fábrica registrada pelo nome lógico do modelo."""
    entry = MODEL_REGISTRY.get(name)
    if not entry or "factory" not in entry:
        raise KeyError(f"Modelo sem fábrica registrada: {name}")
    return globals()[entry["factory"]]
