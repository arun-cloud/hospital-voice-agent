# Hospital Voice AI Agent

## Project overview

This project implements a hospital voice AI agent.

Main components:

- Python LiveKit voice agent
- C# ASP.NET backend API
- Patient verification using mobile number and patient ID
- English and Arabic voice interaction
- LiveKit Cloud used for voice sessions
- Mock patient data currently used by backend API

## Architecture

Voice flow:

User
-> LiveKit
-> Python Agent
-> Backend API
-> Patient data / mock data
-> Agent response
-> LiveKit voice output

## Development rules

Before modifying code:

1. Inspect the relevant files.
2. Trace the execution flow.
3. Identify the root cause.
4. Explain the proposed change.
5. Modify only files necessary for the task.

Do not guess about code that has not been inspected.

Do not perform unrelated refactoring.

Do not change public APIs unless required.

Do not hard-code test values.

Do not bypass authentication or validation just to make tests pass.

## Python

- Follow existing project structure.
- Use async correctly.
- Handle API failures explicitly.
- Preserve LiveKit agent lifecycle behavior.
- Log important workflow transitions.
- Never log secrets or patient-sensitive information.

## C#

- Follow existing ASP.NET architecture.
- Keep controllers thin.
- Business logic belongs in services.
- Use dependency injection.
- Preserve API response contracts unless explicitly changing them.

## Voice agent workflow

Important states include:

1. Agent startup
2. Language selection
3. Mobile number collection
4. Patient ID collection
5. Backend verification
6. Patient confirmation
7. Hospital service workflow

When debugging the agent, trace state transitions rather than modifying prompts blindly.

## Arabic

When Arabic is selected:

- maintain the same business workflow as English
- only language-specific prompts should differ
- API/business logic should not diverge unless explicitly required

## Verification

Before considering a task complete:

- run relevant Python tests
- run relevant .NET tests
- check logs for exceptions
- verify the intended conversation flow
- inspect git diff for unrelated changes

## Security

Never commit:

- API keys
- LiveKit secrets
- access tokens
- patient data
- credentials

Use environment variables or local configuration excluded from Git.