"""Agent authentication for the support-agent API.

Authorization model (deliberate, not accidental): support agents in this
domain legitimately need to look up *any* customer's record -- there is no
"agent owns customer X" relationship anywhere in the data
(`data/structured/customers.csv` has no agent-assignment field), so this
module does not attempt row-level ownership checks. Instead it follows the
same shape real support tools (Zendesk, Salesforce Service Cloud) use:

1. Authenticate every request -- prove the caller is a real, provisioned
   agent, not an anonymous caller hitting the API directly.
2. Authorize broadly -- any authenticated agent may query any customer.
   That's the job; scoping it further would be inventing a constraint the
   data doesn't support.
3. Audit-log every access (see `app/observability.py`) so who-looked-up-what
   is traceable after the fact, even though nothing is blocked at query time.

Keys live in an environment variable (name configurable via
`config.auth.keys_env`, default `AGENT_API_KEYS` -- see `app/config.py`),
never in code or YAML, matching every other `*_env` secret/endpoint field in
this project's config.

Like `app/security/sanitizer.py`'s `HeuristicSanitizer`, this whole feature is
gated by a single config flag (`config.auth.enabled`) that fully no-ops it:
when disabled, `require_agent` returns a fixed anonymous principal instead of
raising, so local dev and tests that don't care about auth mechanics aren't
forced to configure keys.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel

from app.config import AppConfig, load_config


class AgentPrincipal(BaseModel):
    """Identifies which support agent made a request.

    Deliberately minimal -- just `agent_id`, no `region`/scoping field. Per
    the authorization model above, any authenticated agent can query any
    customer, so there is nothing else to encode here; adding a scoping
    field would imply a per-agent restriction this domain doesn't have.
    """

    agent_id: str


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Cached `AppConfig` accessor used as a FastAPI dependency by
    `require_agent` below.

    Defined here (rather than duplicated as a second `@lru_cache` in
    `app/api.py`) so there is exactly one cached config instance shared by
    the whole DI graph -- `app/api.py` imports this same symbol rather than
    keeping its own cache that could silently disagree with this module's
    after a `dependency_overrides` swap in tests.
    """
    return load_config()


def parse_agent_keys(raw: str) -> dict[str, str]:
    """Parse the `AGENT_API_KEYS`-style environment variable into a
    `{key: agent_id}` lookup map.

    Format: comma-separated `key:agent_id` pairs, e.g.

        AGENT_API_KEYS=sk-abc123:agent_alice,sk-def456:agent_bob

    Each pair is `<opaque bearer token>:<agent_id>`. Whitespace around
    commas/colons is tolerated (so a human editing a `.env` file by hand
    doesn't have to be exact). Empty segments (from a stray trailing comma,
    or an unset/empty env var) are silently skipped rather than raising --
    this function is also called with `""` when the env var is unset, and
    should just yield an empty map in that case rather than erroring, since
    "no keys configured" is a valid (if locked-down) state distinct from
    `auth.enabled=False`. A segment missing the `:` separator, or with an
    empty key or empty agent_id, is likewise skipped -- malformed config
    should not crash the process, it should just mean that entry never
    matches any request.
    """
    keys: dict[str, str] = {}
    if not raw:
        return keys
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, _, agent_id = pair.partition(":")
        key = key.strip()
        agent_id = agent_id.strip()
        if key and agent_id:
            keys[key] = agent_id
    return keys


def _load_agent_keys(config: AppConfig) -> dict[str, str]:
    """Re-reads and re-parses the env var on every call (no caching) so
    tests can monkeypatch/override the key set without fighting a stale
    cache -- parsing a short comma-separated string is cheap enough that
    doing it once per request is not worth the staleness risk."""
    raw = os.environ.get(config.auth.keys_env, "")
    return parse_agent_keys(raw)


def require_agent(
    authorization: Optional[str] = Header(None),
    config: AppConfig = Depends(get_config),
) -> AgentPrincipal:
    """FastAPI dependency: authenticates the caller as a known support agent.

    Expects `Authorization: Bearer <key>`. Raises `HTTPException(401)` on a
    missing header, a header that isn't shaped like `Bearer <token>`, or a
    key not present in the parsed `AGENT_API_KEYS` map. Returns the matching
    `AgentPrincipal` on success.

    When `config.auth.enabled` is `False`, this fully no-ops (see module
    docstring): it never raises and returns a fixed
    `AgentPrincipal(agent_id="anonymous")` regardless of what -- if
    anything -- was sent in the `Authorization` header.
    """
    if not config.auth.enabled:
        return AgentPrincipal(agent_id="anonymous")

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401, detail="Authorization header must be 'Bearer <key>'"
        )

    agent_keys = _load_agent_keys(config)
    agent_id = agent_keys.get(token)
    if agent_id is None:
        raise HTTPException(status_code=401, detail="Unknown API key")

    return AgentPrincipal(agent_id=agent_id)
