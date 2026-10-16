"""Domain errors shared by the API and offline workflow."""


class SupportIQError(Exception):
    """Base exception with a stable public error code."""

    code = "supportiq_error"


class InvalidMessageError(SupportIQError, ValueError):
    """Input text is empty or has an invalid type."""

    code = "invalid_message"


class DatasetError(SupportIQError, ValueError):
    """A labeled dataset violates the training contract."""

    code = "invalid_dataset"


class ArtifactError(SupportIQError):
    """The configured trained model cannot be loaded or used."""

    code = "model_unavailable"


class InferenceError(SupportIQError):
    """The trained model failed to produce valid probabilities."""

    code = "inference_failed"


class BatchSizeError(SupportIQError, ValueError):
    """The batch exceeds the configured size limit."""

    code = "invalid_batch_size"
