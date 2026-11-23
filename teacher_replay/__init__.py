"""Teacher Replay: effective and verifiable auditing of prompt use in distilled language models.

Modules
* :mod:`teacher_replay.owner` -- random split, hash commitment, strata (no torch).
* :mod:`teacher_replay.replay` -- teacher replays at the training recipe (vLLM or Hugging Face).
* :mod:`teacher_replay.shadow` -- prompt-masked SFT of shadow students (TRL).
* :mod:`teacher_replay.score` -- teacher-forced per-token log-likelihoods.
* :mod:`teacher_replay.audit` -- TR / TR-L / TR-M and the exact owner-level test (no torch).
"""

__version__ = "0.1.0"
