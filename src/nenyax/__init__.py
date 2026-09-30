"""Nenyax: one contract for every RL environment::

    import nenyax

    env = nenyax.load("reasoning_gym:basic_arithmetic?size=10")
    traj = env.rollout(nenyax.Endpoint("http://localhost:8000/v1", "Qwen/Qwen3-8B"))
    print(traj.score)

See README.md and docs/protocol.md.
"""

from . import config, judges, learners
from .conformance import Report, Tier, check
from .define import define
from .environment import (
    Environment,
    ModelDrivenEnvironment,
    NenyaxError,
    Policy,
    PolicyMismatch,
    Session,
    StepEnvironment,
    UnsupportedCapability,
)
from .judges import with_judge
from .policy import ChatPolicy, Endpoint, FunctionModel, NativeAgent, chat_policy
from .registry import Driver, DriverNotFound, DriverUnavailable, drivers, load, register
from .runner import run
from .server import ModelServer
from .train import TrainingError, TrainResult, train
from .types import (
    PROTOCOL_VERSION,
    Capabilities,
    Isolation,
    Judgment,
    Manifest,
    Message,
    Mode,
    ModelCall,
    Observation,
    Step,
    TaskRef,
    ToolCall,
    Trajectory,
)

__version__ = "0.1.0a2"

__all__ = [
    "Capabilities",
    "ChatPolicy",
    "Driver",
    "DriverNotFound",
    "DriverUnavailable",
    "Endpoint",
    "Environment",
    "FunctionModel",
    "Isolation",
    "Judgment",
    "Manifest",
    "Message",
    "Mode",
    "ModelCall",
    "ModelDrivenEnvironment",
    "ModelServer",
    "NativeAgent",
    "NenyaxError",
    "Observation",
    "PROTOCOL_VERSION",
    "Policy",
    "PolicyMismatch",
    "Report",
    "Session",
    "Step",
    "StepEnvironment",
    "TaskRef",
    "Tier",
    "ToolCall",
    "Trajectory",
    "UnsupportedCapability",
    "chat_policy",
    "config",
    "define",
    "judges",
    "learners",
    "train",
    "TrainResult",
    "TrainingError",
    "with_judge",
    "check",
    "drivers",
    "load",
    "register",
    "run",
]
