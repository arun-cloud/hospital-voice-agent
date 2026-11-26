"""Debugger-only wrapper: exposes agent.py's entrypoint as an AgentServer
so `lk agent debugger` can discover it. Does not modify the project."""
import os
import sys

PROJECT = r"D:\My-Projects\hospital-voice-agent\hospital-voice-agent-frontend"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(PROJECT, ".env"))

from livekit.agents import AgentServer  # noqa: E402
from agent import entrypoint  # noqa: E402

server = AgentServer()
server.rtc_session(entrypoint, agent_name="hospital-voice-agent")
