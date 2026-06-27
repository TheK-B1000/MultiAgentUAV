from __future__ import annotations


class CheckpointError(RuntimeError):
    def __init__(self, message: str, **context: object) -> None:
        self.context = {k: v for k, v in context.items() if v is not None}
        if self.context:
            detail = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            message = f"{message} ({detail})"
        super().__init__(message)


class CheckpointNotFoundError(CheckpointError):
    pass


class CheckpointArchiveError(CheckpointError):
    pass


class CheckpointMetadataError(CheckpointError):
    pass


class CheckpointSchemaError(CheckpointError):
    pass


class CheckpointStateDictError(CheckpointError):
    pass


class CheckpointCompatibilityError(CheckpointError):
    pass


class UnsupportedCheckpointMigrationError(CheckpointCompatibilityError):
    pass


class CheckpointBehavioralEquivalenceError(CheckpointCompatibilityError):
    pass


class CheckpointModelConstructionError(CheckpointError):
    pass
