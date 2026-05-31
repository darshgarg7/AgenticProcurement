"""Episode-level metric accumulation."""

from typing import Any


class MetricsTracker:
    """Track query, delay, purchase, and regret statistics for one episode."""

    def __init__(self) -> None:
        """Initialize all counters and regret fields."""
        self.queries: int = 0
        self.delays: int = 0
        self.purchased: bool = False
        self.realized_regret: float = 0.0
        self.estimated_worst_case_regret: float = 0.0
        self.exceeded_regret: bool = False
        
    def record_query(self) -> None:
        """Record one user-query action."""
        self.queries += 1
        
    def record_delay(self) -> None:
        """Record one non-query delay action, such as wait or search."""
        self.delays += 1
        
    def record_purchase(self, realized_regret: float, estimated_wc_regret: float, threshold: float = 1.0) -> None:
        """Record the terminal purchase outcome."""
        self.purchased = True
        self.realized_regret = realized_regret
        self.estimated_worst_case_regret = estimated_wc_regret
        self.exceeded_regret = realized_regret > threshold
        
    def get_stats(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics dictionary."""
        return {
            'queries': self.queries,
            'delays': self.delays,
            'purchased': self.purchased,
            'realized_regret': self.realized_regret,
            'estimated_worst_case_regret': self.estimated_worst_case_regret,
            'exceeded_regret': self.exceeded_regret
        }
