"""
Routing Pattern

This module demonstrates the Routing design pattern - classifying incoming
queries and routing them to specialized handlers based on the classification.

Pattern Overview:
1. Classify: Determine the category/route for an incoming query
2. Route: Send query to the appropriate specialized handler
3. Handle: Process query with domain-specific logic
4. Respond: Return appropriate response

When to Use:
- When you have distinct types of queries that need different handling
- When you want specialized agents for different domains (e.g., tech support vs sales)
- When classification can be done reliably before processing
- When different routes require different expertise or tools

When NOT to Use:
- When classification is unreliable or ambiguous
- When queries need multiple handlers simultaneously
- When the routing logic is too complex (use Orchestrator pattern instead)
- When you need the agent to decide dynamically (use ReAct pattern instead)
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field
from utils.llm import LLMClient, call_llm
from utils.agent_logging import logger
from utils.config import config
from utils.schemas import RouteClassification


# =============================================================================
# Pydantic Models for Routing
# =============================================================================

class QueryClassification(BaseModel):
    """
    Result of query classification.
    
    This determines which route/handler should process the query.
    """
    route: Literal["tech_support", "sales", "general", "billing"] = Field(
        description="The route category for this query"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) for this classification"
    )
    reasoning: str = Field(
        description="Explanation of why this route was chosen"
    )


class TechSupportResponse(BaseModel):
    """Response from tech support handler."""
    response: str = Field(description="Technical support response")
    troubleshooting_steps: list[str] = Field(
        default_factory=list,
        description="Step-by-step troubleshooting instructions"
    )
    escalation_needed: bool = Field(
        default=False,
        description="Whether this needs to be escalated to human support"
    )
    related_topics: list[str] = Field(
        default_factory=list,
        description="Related topics or documentation links"
    )


class SalesResponse(BaseModel):
    """Response from sales handler."""
    response: str = Field(description="Sales response")
    product_recommendations: list[str] = Field(
        default_factory=list,
        description="Recommended products or services"
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Suggested next steps for the customer"
    )
    estimated_value: Optional[str] = Field(
        default=None,
        description="Estimated value or pricing information"
    )


class GeneralResponse(BaseModel):
    """Response from general handler (fallback)."""
    response: str = Field(description="General response")
    suggested_route: Optional[str] = Field(
        default=None,
        description="Suggested route if query might fit better elsewhere"
    )


class BillingResponse(BaseModel):
    """Response from billing handler."""
    response: str = Field(description="Billing-related response")
    account_info_summary: Optional[str] = Field(
        default=None,
        description="Summary of account information if relevant"
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Action items or steps to resolve billing issue"
    )


# =============================================================================
# Routing Implementation
# =============================================================================

class QueryRouter:
    """
    Implements the Routing pattern for query classification and handling.
    
    This class:
    1. Classifies incoming queries into categories
    2. Routes queries to specialized handlers
    3. Returns appropriate responses based on the route
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the query router.
        
        Args:
            provider: LLM provider to use (defaults to config)
        """
        self.client = LLMClient(provider=provider)
        self.confidence_threshold = config.get_nested(
            "patterns.routing.confidence_threshold", 
            0.7
        )
        self.fallback_to_general = config.get_nested(
            "patterns.routing.fallback_to_general",
            True
        )
    
    def route_query(self, query: str) -> dict:
        """
        Main entry point: classify and route a query.
        
        Args:
            query: The user's query/question
            
        Returns:
            Dictionary containing the response and metadata
        """
        logger.section("Query Routing", f"Processing query: {query[:50]}...")
        
        # Step 1: Classify the query
        logger.info("Step 1: Classifying query...")
        classification = self._classify_query(query)
        
        logger.action(
            f"Query classified as: {classification.route}",
            arguments={"confidence": classification.confidence}
        )
        logger.observation(
            f"Classification reasoning: {classification.reasoning}"
        )
        
        # Step 2: Route to appropriate handler
        logger.info(f"Step 2: Routing to {classification.route} handler...")
        
        # Check confidence threshold
        if classification.confidence < self.confidence_threshold:
            logger.warning(
                f"Low confidence ({classification.confidence:.2f}) "
                f"below threshold ({self.confidence_threshold})"
            )
            if self.fallback_to_general:
                logger.info("Falling back to general handler")
                classification.route = "general"
        
        # Route to handler
        response = self._route_to_handler(query, classification.route)
        
        logger.success(f"Query processed successfully via {classification.route} route")
        
        return {
            "query": query,
            "classification": classification.model_dump(),
            "response": response.model_dump(),
            "route_used": classification.route
        }
    
    def _classify_query(self, query: str) -> QueryClassification:
        """
        Classify a query into one of the available routes.
        
        Args:
            query: The user's query
            
        Returns:
            Classification result with route and confidence
        """
        system_prompt = """You are an expert at classifying customer queries.
Analyze the query and determine which department should handle it.

Available routes:
- tech_support: Technical issues, bugs, how-to questions, troubleshooting
- sales: Product inquiries, pricing, features, purchasing decisions
- billing: Payment issues, invoices, refunds, account billing questions
- general: General questions that don't fit other categories

Be confident in your classification. Provide reasoning for your choice."""
        
        user_prompt = f"""Classify this customer query:

"{query}"

Determine:
1. Which route should handle this query (tech_support, sales, billing, or general)
2. Your confidence level (0.0 to 1.0)
3. Brief reasoning for your classification"""
        
        logger.thought(f"Analyzing query to determine appropriate route")
        
        try:
            classification = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                response_model=QueryClassification
            )
            
            return classification
            
        except Exception as e:
            logger.error(f"Classification failed: {e}", exception=e)
            # Fallback to general if classification fails
            return QueryClassification(
                route="general",
                confidence=0.5,
                reasoning=f"Classification failed, defaulting to general: {str(e)}"
            )
    
    def _route_to_handler(
        self, 
        query: str, 
        route: str
    ) -> BaseModel:
        """
        Route query to the appropriate specialized handler.
        
        Args:
            query: The user's query
            route: The route category
            
        Returns:
            Response from the handler
        """
        handlers = {
            "tech_support": self._handle_tech_support,
            "sales": self._handle_sales,
            "billing": self._handle_billing,
            "general": self._handle_general
        }
        
        handler = handlers.get(route, self._handle_general)
        return handler(query)
    
    def _handle_tech_support(self, query: str) -> TechSupportResponse:
        """
        Handle technical support queries.
        
        Args:
            query: The technical support query
            
        Returns:
            Tech support response
        """
        system_prompt = """You are an expert technical support agent.
Provide helpful, clear technical assistance. Include troubleshooting steps
when applicable. Escalate to human support if the issue is complex or requires
account access."""
        
        user_prompt = f"""A customer has submitted this technical support query:

"{query}"

Provide:
1. A helpful response addressing their issue
2. Step-by-step troubleshooting instructions if applicable
3. Whether this needs escalation to human support
4. Related topics or documentation that might help"""
        
        logger.thought("Processing query with tech support expertise")
        
        try:
            response = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                response_model=TechSupportResponse
            )
            
            logger.observation(
                f"Tech support response generated. "
                f"Escalation needed: {response.escalation_needed}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Tech support handler failed: {e}", exception=e)
            raise
    
    def _handle_sales(self, query: str) -> SalesResponse:
        """
        Handle sales queries.
        
        Args:
            query: The sales query
            
        Returns:
            Sales response
        """
        system_prompt = """You are an expert sales agent.
Help customers understand products, features, and pricing. Provide
product recommendations based on their needs. Guide them toward
the next steps in the sales process."""
        
        user_prompt = f"""A customer has submitted this sales inquiry:

"{query}"

Provide:
1. A helpful sales response addressing their needs
2. Product or service recommendations
3. Suggested next steps for the customer
4. Pricing or value information if relevant"""
        
        logger.thought("Processing query with sales expertise")
        
        try:
            response = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                response_model=SalesResponse
            )
            
            logger.observation(
                f"Sales response generated. "
                f"Recommended {len(response.product_recommendations)} products"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Sales handler failed: {e}", exception=e)
            raise
    
    def _handle_billing(self, query: str) -> BillingResponse:
        """
        Handle billing queries.
        
        Args:
            query: The billing query
            
        Returns:
            Billing response
        """
        system_prompt = """You are a billing support agent.
Help customers with payment issues, invoices, refunds, and account billing.
Be clear and helpful. Provide action items when applicable."""
        
        user_prompt = f"""A customer has submitted this billing inquiry:

"{query}"

Provide:
1. A helpful response addressing their billing question
2. Summary of relevant account information if applicable
3. Clear action items or steps to resolve the issue"""
        
        logger.thought("Processing query with billing expertise")
        
        try:
            response = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                response_model=BillingResponse
            )
            
            logger.observation(
                f"Billing response generated. "
                f"{len(response.action_items)} action items provided"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Billing handler failed: {e}", exception=e)
            raise
    
    def _handle_general(self, query: str) -> GeneralResponse:
        """
        Handle general queries (fallback handler).
        
        Args:
            query: The general query
            
        Returns:
            General response
        """
        system_prompt = """You are a helpful customer service agent.
Answer general questions helpfully. If the query might fit better
in another department, suggest that route."""
        
        user_prompt = f"""A customer has submitted this query:

"{query}"

Provide:
1. A helpful general response
2. If applicable, suggest which department might better handle this query"""
        
        logger.thought("Processing query with general handler")
        
        try:
            response = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                response_model=GeneralResponse
            )
            
            logger.observation("General response generated")
            
            return response
            
        except Exception as e:
            logger.error(f"General handler failed: {e}", exception=e)
            raise


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Routing Pattern Demo")
    print("=" * 60)
    
    router = QueryRouter()
    
    # Test queries
    test_queries = [
        "I can't log into my account. It says password incorrect.",
        "What are your pricing plans? I'm interested in the Pro tier.",
        "I was charged twice for my subscription. Can I get a refund?",
        "What is your company's return policy?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        try:
            result = router.route_query(query)
            
            print(f"\nRoute: {result['route_used']}")
            print(f"Confidence: {result['classification']['confidence']:.2f}")
            print(f"\nResponse:")
            print(result['response']['response'])
            
            # Show route-specific details
            if result['route_used'] == 'tech_support':
                if result['response'].get('troubleshooting_steps'):
                    print("\nTroubleshooting Steps:")
                    for step in result['response']['troubleshooting_steps']:
                        print(f"  - {step}")
            
            elif result['route_used'] == 'sales':
                if result['response'].get('product_recommendations'):
                    print("\nProduct Recommendations:")
                    for product in result['response']['product_recommendations']:
                        print(f"  - {product}")
            
            elif result['route_used'] == 'billing':
                if result['response'].get('action_items'):
                    print("\nAction Items:")
                    for item in result['response']['action_items']:
                        print(f"  - {item}")
        
        except Exception as e:
            print(f"\nError: {e}")
            print("\nNote: Make sure you have:")
            print("1. Copied .env.example to .env")
            print("2. Added your API key to .env")
            print("3. Installed all dependencies: pip install -r requirements.txt")



