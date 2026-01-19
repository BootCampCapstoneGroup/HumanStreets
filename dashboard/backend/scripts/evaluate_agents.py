"""
Agent Evaluation Script
Run standalone to test agent routing and responses.

Usage:
    cd backend
    uv run python evaluate_agents.py
"""

import asyncio
import sys
from app.agents.router import RouterAgent
from app.services.llm import llm_service
from app.services.spatial import spatial_service

# Initialize services
print("Initializing Services for Test...")
try:
    llm_service.initialize_models()
    spatial_service.load_data()
except Exception as e:
    print(f"Service Init Warning (might be fine if data missing locally): {e}")


class TestResult:
    def __init__(self, name: str, passed: bool, response: str, details: str = ""):
        self.name = name
        self.passed = passed
        self.response = response
        self.details = details


async def run_test(router: RouterAgent, name: str, query: str, expected_keywords: list[str], 
                   negative_keywords: list[str] = None, any_of: list[str] = None,
                   provider: str = "gemini") -> TestResult:
    """Run a single test case and return result."""
    print(f"\n{'='*60}")
    print(f"📋 Test: {name}")
    print(f"   Query: {query}")
    print(f"   Expected: {expected_keywords}")
    if any_of:
        print(f"   Any of: {any_of}")
    
    response_text = ""
    try:
        async for chunk in router.route_and_execute(query, history=[], provider=provider):
            response_text += chunk
    except Exception as e:
        return TestResult(name, False, str(e), f"Exception: {e}")
    
    # Check expected keywords (ALL must be present)
    all_found = all(kw.lower() in response_text.lower() for kw in expected_keywords)
    
    # Check any_of keywords (at least ONE must be present)
    any_found = True
    if any_of:
        any_found = any(kw.lower() in response_text.lower() for kw in any_of)
    
    # Check negative keywords (should NOT be present)
    no_negatives = True
    if negative_keywords:
        no_negatives = not any(kw.lower() in response_text.lower() for kw in negative_keywords)
    
    passed = all_found and any_found and no_negatives
    
    details = ""
    if not all_found:
        missing = [kw for kw in expected_keywords if kw.lower() not in response_text.lower()]
        details += f"Missing keywords: {missing}. "
    if not any_found:
        details += f"None of {any_of} found. "
    if not no_negatives and negative_keywords:
        found_neg = [kw for kw in negative_keywords if kw.lower() in response_text.lower()]
        details += f"Unwanted keywords found: {found_neg}. "
    
    print(f"   Response preview: {response_text[:200]}...")
    print(f"   {'✅ PASS' if passed else '❌ FAIL'}" + (f" - {details}" if details else ""))
    
    return TestResult(name, passed, response_text, details)


async def main():
    router = RouterAgent()
    results = []

    # ===========================================
    # TEST SUITE: Neighborhood Lookup (Arabic)
    # ===========================================
    
    # Test 1: Simple Arabic neighborhood lookup with حي prefix
    # Expected: Routes to LOOKUP, generates SQL with ILIKE and geometry AS geom
    results.append(await run_test(
        router,
        "Arabic Neighborhood Lookup - حي النرجس",
        "show me the walkability for حي النرجس",
        expected_keywords=["SELECT", "النرجس"],
        any_of=["geom", "geometry"],  # SQL might use either
        negative_keywords=["Plan Proposed"]  # Should NOT trigger planner
    ))

    # Test 2: Different phrasing in Arabic
    results.append(await run_test(
        router,
        "Arabic Neighborhood Lookup - اعرض حي",
        "اعرض حي الروضة",
        expected_keywords=["SELECT", "الروضة"],
        any_of=["geom", "geometry"],
        negative_keywords=["Plan Proposed"]
    ))

    # ===========================================
    # TEST SUITE: Layer Operations
    # ===========================================
    
    # Test 3: Districts -> Neighborhoods alias
    results.append(await run_test(
        router,
        "Layer Alias - Districts to Neighborhoods",
        "Show districts layer",
        expected_keywords=["NEIGHBORHOODS"],
    ))

    # ===========================================
    # TEST SUITE: Complex Queries -> PLANNER
    # ===========================================
    
    # Test 4: Complex query SHOULD trigger planner (has 'where' and 'best')
    results.append(await run_test(
        router,
        "Complex Query - Eastern + Best (PLANNER)",
        "Show me the eastern neighborhoods where the walkability score is best",
        expected_keywords=["Plan Proposed"],
    ))

    # Test 5: Query with 'above' filter
    results.append(await run_test(
        router,
        "Complex Query - Above Threshold (PLANNER)",
        "Find neighborhoods with walkability score above 60",
        expected_keywords=["Plan Proposed"],
    ))

    # ===========================================
    # TEST SUITE: Simple Stats Query -> ANALYTICS
    # ===========================================
    
    # Test 6: Simple count query (should use ANALYTICS/SQL)
    results.append(await run_test(
        router,
        "Analytics Query - Count",
        "How many neighborhoods are there in total?",
        expected_keywords=["SELECT", "COUNT"],
        negative_keywords=["Plan Proposed"]
    ))

    # ===========================================
    # SUMMARY
    # ===========================================
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.name}")
        if not r.passed and r.details:
            print(f"      └── {r.details}")
    
    print(f"\n  Total: {passed}/{len(results)} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
