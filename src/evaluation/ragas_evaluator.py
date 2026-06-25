"""
RAGAS evaluation engine.

WHY: Enterprise RAG platforms must systematically measure output quality. This module
runs asynchronous evaluation loops against golden datasets, calculates faithfulness,
relevance, and retrieval quality using Ragas, and records runs to PostgreSQL.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from sqlalchemy import update

from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.evaluation.datasets.golden_dataset import get_dataset
from src.memory.models import EvaluationResult, EvaluationRun

logger = get_logger(__name__)

# Map string names to pre-instantiated Ragas metrics
METRICS_MAP = {
    "faithfulness": faithfulness,
    "answer_relevance": answer_relevancy,
    "context_recall": context_recall,
    "context_precision": context_precision,
}


class RagasEvaluator:
    """
    Orchestrates Ragas evaluations against golden datasets.
    
    Generates answers from the current agent configuration, executes
    Ragas metrics via the gateway, and saves detailed telemetry in PostgreSQL.
    """

    def __init__(
        self,
        settings: Settings,
        graph: Any,
        memory_store: Any,
        graph_config: Dict[str, Any],
    ) -> None:
        self.settings = settings
        self.graph = graph
        self.memory_store = memory_store
        self.graph_config = graph_config

    async def run_evaluation(
        self,
        run_id: str,
        dataset_name: str,
        questions: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Execute evaluation asynchronously.
        
        Saves a pending run, generates predictions using the agent graph,
        runs Ragas evaluations, and persists the final scores.
        """
        start_time = time.perf_counter()
        session_factory = self.memory_store.session_factory

        # Resolve metrics
        metrics_list = metrics or list(METRICS_MAP.keys())
        ragas_metrics = [METRICS_MAP[m] for m in metrics_list if m in METRICS_MAP]

        # 1. Create a pending run row in the DB
        async with session_factory() as db:
            run_uuid = uuid.UUID(run_id)
            run = EvaluationRun(
                id=run_uuid,
                name=f"Run_{dataset_name}_{run_id[:8]}",
                dataset_name=dataset_name,
                status="running",
                metrics={},
                config={
                    "metrics_requested": metrics_list,
                    "model": self.settings.llm.primary_model,
                },
            )
            db.add(run)
            await db.commit()

        logger.info("Evaluation run started in background", run_id=run_id, dataset=dataset_name)

        try:
            # 2. Load golden dataset samples
            samples = get_dataset(dataset_name)
            if questions:
                samples = [s for s in samples if s["question"] in questions]

            if not samples:
                raise ValueError(f"No evaluation samples found for dataset: {dataset_name}")

            # 3. Generate answers from the LangGraph Agent
            from src.agent.graph import run_agent

            eval_dataset_inputs = []
            
            for idx, sample in enumerate(samples):
                logger.info(
                    "Generating answer for evaluation sample",
                    run_id=run_id,
                    index=idx + 1,
                    total=len(samples),
                )
                
                # Execute agent query
                result = await run_agent(
                    graph=self.graph,
                    query=sample["question"],
                    session_id=f"eval_{run_id}_{idx}",
                    user_id="evaluator",
                    config=self.graph_config,
                )

                # Format retrieved contexts (reconstruct page_content from Document objects)
                docs = result.get("reranked_documents", []) or result.get("documents", [])
                contexts = [doc.page_content for doc in docs]

                eval_dataset_inputs.append(
                    {
                        "question": sample["question"],
                        "ground_truth": sample["ground_truth"],
                        "answer": result.get("response", ""),
                        "contexts": contexts,
                    }
                )

            # 4. Convert generated predictions to Hugging Face Dataset for Ragas
            dataset_dict = {
                "question": [s["question"] for s in eval_dataset_inputs],
                "contexts": [s["contexts"] for s in eval_dataset_inputs],
                "answer": [s["answer"] for s in eval_dataset_inputs],
                "ground_truth": [s["ground_truth"] for s in eval_dataset_inputs],
            }
            hf_dataset = Dataset.from_dict(dataset_dict)

            # 5. Build LLM and Embeddings wrappers for Ragas evaluation
            # Reuses our Portkey client & FastEmbed managers to avoid API config mismatch
            gateway_client = self.graph_config["gateway"]._build_client()
            embedding_client = self.graph_config["vectorstore"].embedding_manager.get_model()

            logger.info("Computing Ragas metrics", run_id=run_id)
            
            # Execute Ragas evaluation
            # Ragas expects an async call loop or synchronous execute
            eval_result = evaluate(
                dataset=hf_dataset,
                metrics=ragas_metrics,
                llm=gateway_client,
                embeddings=embedding_client,
            )

            # 6. Parse results and update DB tables
            df_results = eval_result.to_pandas()
            duration = time.perf_counter() - start_time

            async with session_factory() as db:
                # Add individual sample results
                for idx, row in df_results.iterrows():
                    res = EvaluationResult(
                        run_id=run_uuid,
                        question=row["question"],
                        ground_truth=row["ground_truth"],
                        generated_answer=row["answer"],
                        contexts=row["contexts"],
                        metrics={m: float(row[m]) for m in metrics_list if m in row},
                    )
                    db.add(res)

                # Update the main run summary
                summary_metrics = {m: float(eval_result[m]) for m in metrics_list if m in eval_result}
                
                await db.execute(
                    update(EvaluationRun)
                    .where(EvaluationRun.id == run_uuid)
                    .values(
                        status="completed",
                        metrics=summary_metrics,
                        num_samples=len(samples),
                        duration_seconds=round(duration, 2),
                    )
                )
                await db.commit()

            logger.info("Evaluation run completed successfully", run_id=run_id, duration_seconds=round(duration, 2))

        except Exception as e:
            logger.error("Evaluation run failed", run_id=run_id, error=str(e))
            duration = time.perf_counter() - start_time
            
            async with session_factory() as db:
                await db.execute(
                    update(EvaluationRun)
                    .where(EvaluationRun.id == run_uuid)
                    .values(
                        status="failed",
                        error=str(e),
                        duration_seconds=round(duration, 2),
                    )
                )
                await db.commit()
