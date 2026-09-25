"""Load graph inputs from Postgres, through the agents' read-only login.

The graph never writes: these loaders only SELECT, using the same SELECT-only role as the tools. Persisting runs
and outcomes is the API layer's job (Phase 7), in deterministic code.
"""

import uuid
from collections.abc import Awaitable, Callable

from langchain_core.tools import BaseTool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.investigator import INVESTIGATOR_TOOL_NAMES
from app.agents.state import ClaimDocument, ClaimInput
from app.db.models import Claim, Customer, Policy
from app.db.readonly import ReadOnlyDatabase
from app.retrieval.retriever import PolicyRetriever
from app.rules.models import RuleContext
from app.services.rule_context import build_rule_context
from app.tools.policy_tools import build_policy_tools
from app.tools.sql_tools import build_sql_tools


class ClaimNotFoundError(LookupError):
    """No claim with the given number or id."""


async def load_claim_input(
    database: ReadOnlyDatabase, claim_number: str, document: ClaimDocument | None = None
) -> ClaimInput:
    """Build the graph input from a stored claim.

    Args:
        database: Read-only database.
        claim_number: The claim to load.
        document: Extracted text of the uploaded PDF, when there is one.

    Returns:
        The claim as the graph sees it.

    Raises:
        ClaimNotFoundError: If the claim does not exist.
    """
    statement = (
        select(Claim, Policy.policy_number, Policy.product_type)
        .join(Policy, Policy.id == Claim.policy_id)
        .where(Claim.claim_number == claim_number)
    )
    async with database.connect() as connection:
        # ORM entities (a Claim object, not flat columns) need a Session; this one is bound to the read-only connection.
        session = AsyncSession(bind=connection)
        try:
            row = (await session.execute(statement)).one_or_none()
        finally:
            await session.close()
    if row is None:
        raise ClaimNotFoundError(claim_number)
    claim: Claim = row.Claim
    return ClaimInput(
        claim_id=claim.id,
        claim_number=claim.claim_number,
        policy_number=row.policy_number,
        product_type=row.product_type,
        incident_type=claim.incident_type,
        incident_date=claim.incident_date,
        claimed_amount=claim.claimed_amount,
        description=claim.description,
        submitted_at=claim.created_at,
        document=document,
    )


async def load_rule_context(database: ReadOnlyDatabase, claim_id: uuid.UUID) -> RuleContext:
    """Load everything the rules engine needs for one claim.

    The rules run on the stored claim (form values), not on what intake extracted: the deterministic core must
    not depend on an LLM's reading of a document. Disagreements show up as intake mismatches instead.

    Args:
        database: Read-only database.
        claim_id: The claim.

    Returns:
        The RuleContext.

    Raises:
        ClaimNotFoundError: If the claim does not exist.
    """
    async with database.connect() as connection:
        # A session bound to the read-only connection, so ORM objects load without any write privilege.
        session = AsyncSession(bind=connection)
        try:
            claim = await session.get(Claim, claim_id)
            if claim is None:
                raise ClaimNotFoundError(str(claim_id))
            policy = await session.get_one(Policy, claim.policy_id)
            customer = await session.get_one(Customer, policy.customer_id)
            customer_claims = (
                await session.scalars(
                    select(Claim).join(Policy, Policy.id == Claim.policy_id).where(Policy.customer_id == customer.id)
                )
            ).all()
        finally:
            await session.close()
    return build_rule_context(claim, policy, customer, customer_claims)


def rule_context_loader(database: ReadOnlyDatabase) -> Callable[[ClaimInput], Awaitable[RuleContext]]:
    """Adapt ``load_rule_context`` to the graph's dependency signature.

    Args:
        database: Read-only database.

    Returns:
        An async function from ClaimInput to RuleContext.
    """

    async def load(claim: ClaimInput) -> RuleContext:
        return await load_rule_context(database, claim.claim_id)

    return load


def build_investigator_tools(retriever: PolicyRetriever, database: ReadOnlyDatabase) -> list[BaseTool]:
    """All 9 investigator tools: 4 over policy wordings, 5 over the claims database.

    Args:
        retriever: Policy retriever (embedding model loaded once by the caller).
        database: Read-only database.

    Returns:
        The tools.

    Raises:
        RuntimeError: If the set of tools differs from the 9 the investigator is specified to have.
    """
    tools = [*build_policy_tools(retriever), *build_sql_tools(database)]
    names = {tool.name for tool in tools}
    if names != INVESTIGATOR_TOOL_NAMES:
        missing, extra = INVESTIGATOR_TOOL_NAMES - names, names - INVESTIGATOR_TOOL_NAMES
        raise RuntimeError(f"investigator tools mismatch: missing {missing}, extra {extra}")
    return tools
