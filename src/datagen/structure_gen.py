"""
Tier 1 Structure Generator.
Uses Data Profiler and Personas to generate high-volume, structurally diverse questions.
"""
import asyncio
import json

from pathlib import Path
from typing import Dict, List, Any

from src.core.model import APILLM
from src.datagen.profiler import DataProfiler
from src.datagen.personas import PERSONA_REGISTRY, Persona


class StructureGenerator:
    """
    Generates Tier 1 questions by:
    1. Profiling the CSV.
    2. Selecting valid Personas based on the profile.
    3. Prompting LLM with Persona + Fact Bundle.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = APILLM(model=model_name)
        self.profiler = DataProfiler()

    async def generate(self, csv_path: str, n_questions_per_persona: int = 5) -> List[Dict[str, Any]]:
        """Run the full generation pipeline for a dataset."""

        
        # 1. Profile
        try:
            profile = self.profiler.analyze(csv_path)
        except Exception as e:
            return []
            
        # 2. Select Personas
        active_personas = self._select_personas(profile)

        
        # 3. Parallel Generation
        tasks = []
        for persona in active_personas:
            tasks.append(self._generate_for_persona(persona, profile, n_questions_per_persona))
            
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_questions = []
        for res in results:
            all_questions.extend(res)
            

        return all_questions

    def _select_personas(self, profile: Dict[str, Any]) -> List[Persona]:
        """Heuristic to select relevant personas based on data characteristics."""
        selected = []
        
        # Always include Auditor (every dataset needs checking)
        selected.append(PERSONA_REGISTRY["auditor"])
        
        # Check for numeric correlations -> Statistician
        if len(profile.get("correlations", [])) > 0:
            selected.append(PERSONA_REGISTRY["statistician"])
            
        # Check for High Cardinality / Categorical -> Segmenter
        # If any column is 'categorical' type in our profile
        has_categorical = any(
            c_info.get("type") == "categorical" 
            for c_info in profile.get("columns", {}).values()
        )
        if has_categorical:
            selected.append(PERSONA_REGISTRY["segmenter"])
            
        # Check for Dates/Time -> Trend Hunter
        # (Our current V1 profiler detects datetime dtypes)
        has_dates = any(
            c_info.get("type") == "datetime" 
            for c_info in profile.get("columns", {}).values()
        )
        if has_dates:
            selected.append(PERSONA_REGISTRY["trend_hunter"])
            
        return selected

    async def _generate_for_persona(
        self, 
        persona: Persona, 
        profile: Dict[str, Any], 
        n: int
    ) -> List[Dict[str, Any]]:
        """Call LLM for a single persona."""
        
        # Summarize profile for context window (don't dump raw JSON if huge)
        profile_summary = self._summarize_profile_for_prompt(profile)
        
        system_prompt = persona.system_prompt_template.format(
            profile_summary=json.dumps(profile_summary, indent=2)
        )
        
        user_prompt = f"""
        Generate {n} diverse questions based on the profile above.
        Output MUST be a valid JSON list of strings.
        Example: ["Question 1", "Question 2"]
        """
        
        try:
            response_content = await self.llm(
                prompt=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Parse JSON output
            # (Basic parsing, in production use strict json mode or Pydantic)
            content = response_content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            questions = json.loads(content)
            
            # Tag metadata
            tagged_questions = []
            for q in questions:
                tagged_questions.append({
                    "question": q,
                    "persona": persona.name,
                    "tier": 1,
                    "source_csv": profile.get("dataset_name")
                })
            return tagged_questions
            
        except Exception as e:
            return []

    def _summarize_profile_for_prompt(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Compress profile to save tokens if needed (currently pass-through)."""
        # In V2, we might truncate long value lists here
        return profile

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    
    gen = StructureGenerator()
    questions = asyncio.run(gen.generate(args.csv))
    print(json.dumps(questions, indent=2))
